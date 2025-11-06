import asyncio
import json
import logging
import os
import signal
import tempfile
from collections.abc import Mapping
from functools import cached_property, lru_cache
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Dict, List, Optional, Union, Final

from dagster import DagsterLogManager, resource, Field
from dagster_meltano.exceptions import MeltanoCommandError

from dagster_meltano.job import Job
from dagster_meltano.schedule import Schedule
from dagster_meltano.utils import Singleton

STDOUT = 1
OUTPUT_LOGGING_OPTIONS: Final = ["STREAM", "BUFFER", "NONE"]


def execute_shell_command(
    shell_command: str,
    output_logging: str,
    log: Union[logging.Logger, DagsterLogManager],
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    log_shell_command: bool = True,
) -> tuple[str, int]:
    """Execute a shell command using subprocess.
    
    This function replicates the functionality of dagster-shell's execute_shell_command
    using standard library utilities.
    
    The dagster-shell reference implementation can be found here:
    https://github.com/dagster-io/dagster/blob/38cc3bbc1b104613748bf67c1fd2d0d2d17acd70/python_modules/libraries/dagster-shell/dagster_shell/utils.py#L135-L189
    
    Args:
        shell_command (str): The shell command to execute
        output_logging (str): The logging mode to use. Supports STREAM, BUFFER, and NONE.
        log (Union[logging.Logger, DagsterLogManager]): Any logger which responds to .info()
        cwd (str, optional): Working directory for the shell command to use.
        env (Dict[str, str], optional): Environment dictionary to pass to subprocess.Popen.
        log_shell_command (bool, optional): Whether to log the shell command before executing it.
        
    Returns:
        Tuple[str, int]: A tuple where the first element is the combined stdout/stderr output 
        and the second element is the return code.
    """
    if output_logging not in OUTPUT_LOGGING_OPTIONS:
        raise Exception(f"Unrecognized output_logging {output_logging}")

    def pre_exec():
        # Restore default signal disposition and invoke setsid
        for sig in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
            if hasattr(signal, sig):
                signal.signal(getattr(signal, sig), signal.SIG_DFL)
        os.setsid()

    # Create a temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp_file:
        tmp_file.write(shell_command)
        tmp_file.flush()
        script_path = tmp_file.name
        
    try:
        if log_shell_command:
            log.info(f"Running command:\n{shell_command}")

        sub_process = None
        try:
            stdout_pipe = PIPE
            stderr_pipe = STDOUT
            if output_logging == "NONE":
                stdout_pipe = stderr_pipe = None

            sub_process = Popen(
                ["bash", script_path],
                stdout=stdout_pipe,
                stderr=stderr_pipe,
                cwd=cwd,
                env=env,
                preexec_fn=pre_exec,
                encoding="UTF-8",
            )

            log.info(f"Command pid: {sub_process.pid}")

            output = ""
            if output_logging == "STREAM":
                assert sub_process.stdout is not None, "Setting stdout=PIPE should always set stdout."
                # Stream back logs as they are emitted
                lines = []
                for line in sub_process.stdout:
                    log.info(line.rstrip())
                    lines.append(line)
                output = "".join(lines)
            elif output_logging == "BUFFER":
                # Collect and buffer all logs, then emit
                output, _ = sub_process.communicate()
                log.info(output)

            sub_process.wait()
            log.info(f"Command exited with return code {sub_process.returncode}")

            return output, sub_process.returncode
        finally:
            # Always terminate subprocess, including in cases where the run is terminated
            if sub_process:
                sub_process.terminate()
    finally:
        # Clean up the temporary script file
        try:
            os.unlink(script_path)
        except OSError:
            pass


class MeltanoResource(metaclass=Singleton):
    def __init__(
        self,
        project_dir: str = None,
        meltano_bin: Optional[str] = "meltano",
        retries: int = 0,
    ):
        self.project_dir = str(project_dir)
        self.meltano_bin = meltano_bin
        self.retries = retries

    @property
    def default_env(self) -> Dict[str, str]:
        """The default environment to use when running Meltano commands.

        Returns:
            Dict[str, str]: The environment variables.
        """
        return {
            "MELTANO_CLI_LOG_CONFIG": str(Path(__file__).parent / "logging.yaml"),
            "DBT_USE_COLORS": "false",
            "NO_COLOR": "1",
            **os.environ.copy(),
        }

    def execute_command(
        self,
        command: str,
        env: Dict[str, str],
        logger: Union[logging.Logger, DagsterLogManager] = logging.Logger,
    ) -> str:
        """Execute a Meltano command.

        Args:
            context (OpExecutionContext): The Dagster execution context.
            command (str): The Meltano command to execute.
            env (Dict[str, str]): The environment variables to inject when executing the command.

        Returns:
            str: The output of the command.
        """
        output, exit_code = execute_shell_command(
            f"{self.meltano_bin} {command}",
            env={**self.default_env, **env},
            output_logging="STREAM",
            log=logger,
            cwd=self.project_dir,
        )

        if exit_code != 0:
            raise MeltanoCommandError(
                f"Command '{command}' failed with exit code {exit_code}"
            )

        return output

    async def load_json_from_cli(self, command: List[str]) -> dict:
        """Use the Meltano CLI to load JSON data.
        Use asyncio to run multiple commands concurrently.

        Args:
            command (List[str]): The Meltano command to execute.

        Returns:
            dict: The processed JSON data.
        """
        # Create the subprocess, redirect the standard output into a pipe
        proc = await asyncio.create_subprocess_exec(
            self.meltano_bin,
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_dir,
        )

        # Wait for the subprocess to finish
        stdout, stderr = await proc.communicate()

        # Try to load the output as JSON
        try:
            return json.loads(stdout)
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Could not process json: {stdout} {stderr}")

    async def gather_meltano_yaml_information(self):
        jobs, schedules = await asyncio.gather(
            self.load_json_from_cli(["job", "list", "--format=json"]),
            self.load_json_from_cli(["schedule", "list", "--format=json"]),
        )

        return jobs, schedules

    @cached_property
    def meltano_yaml(self) -> dict:
        """Asynchronously load the Meltano jobs and schedules.

        Returns:
            dict: The Meltano jobs and schedules.
        """
        jobs, schedules = asyncio.run(self.gather_meltano_yaml_information())
        return {"jobs": jobs["jobs"], "schedules": schedules["schedules"]}

    @cached_property
    def meltano_jobs(self) -> List[Job]:
        meltano_job_list = self.meltano_yaml["jobs"]
        return [
            Job(
                meltano_job=meltano_job,
                retries=self.retries,
            )
            for meltano_job in meltano_job_list
        ]

    @cached_property
    def meltano_schedules(self) -> List[Schedule]:
        meltano_schedule_list = self.meltano_yaml["schedules"]["job"]
        schedule_list = [
            Schedule(meltano_schedule) for meltano_schedule in meltano_schedule_list
        ]
        return schedule_list

    @property
    def meltano_job_schedules(self) -> Dict[str, Schedule]:
        return {schedule.job_name: schedule for schedule in self.meltano_schedules}

    @property
    def jobs(self) -> List[dict]:
        for meltano_job in self.meltano_jobs:
            yield meltano_job.dagster_job

        for meltano_schedule in self.meltano_schedules:
            yield meltano_schedule.dagster_schedule


@resource(
    description="A resource that corresponds to a Meltano project.",
    config_schema={
        "project_dir": Field(
            str,
            description="The path to the Meltano project.",
            default_value=os.getenv("MELTANO_PROJECT_ROOT", os.getcwd()),
            is_required=False,
        ),
        "retries": Field(
            int,
            description="The number of times to retry a failed job.",
            default_value=0,
            is_required=False,
        ),
    },
)
def meltano_resource(init_context):
    project_dir = init_context.resource_config["project_dir"]
    retries = init_context.resource_config["retries"]

    return MeltanoResource(
        project_dir=project_dir,
        retries=retries,
    )


if __name__ == "__main__":
    meltano_resource = MeltanoResource("/workspace/meltano")
    print(list(meltano_resource.jobs))
    print(meltano_resource.jobs)