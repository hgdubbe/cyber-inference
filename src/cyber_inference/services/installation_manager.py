"""
Installation manager for Cyber-Inference binaries.

Provides:
- Unified interface for llama.cpp and whisper.cpp installations
- Installation status tracking
- System requirements checking
- Progress logging for real-time updates
"""

import asyncio
import shutil
from pathlib import Path
from typing import Literal, Optional

from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger
from cyber_inference.services.llama_installer import LlamaInstaller
from cyber_inference.services.whisper_installer import WhisperInstaller

logger = get_logger(__name__)


class InstallationManager:
    """
    Manages installation and updates for Cyber-Inference binaries.

    Coordinates llama.cpp and whisper.cpp installers and provides
    a unified interface for installation operations.
    """

    def __init__(self) -> None:
        """Initialize the installation manager."""
        settings = get_settings()
        self.bin_dir = settings.bin_dir
        self.llama_installer = LlamaInstaller(bin_dir=self.bin_dir)
        self.whisper_installer = WhisperInstaller(bin_dir=self.bin_dir)
        logger.info("InstallationManager initialized")

    async def get_installation_status(self) -> dict:
        """
        Get installation status for both binaries.

        Returns:
            Dict with status for llama and whisper, plus system info
        """
        logger.debug("Checking installation status for both binaries")

        # Get llama status
        llama_installed = self.llama_installer.is_installed()
        llama_version = None
        llama_path = None
        if llama_installed:
            llama_version = await self.llama_installer.get_installed_version()
            llama_path = str(self.llama_installer.get_binary_path())

        # Get whisper status
        whisper_installed = self.whisper_installer.is_installed()
        whisper_version = None
        whisper_path = None
        if whisper_installed:
            whisper_version = await self.whisper_installer.get_installed_version()
            whisper_path = str(self.whisper_installer.get_binary_path())

        # Get system requirements
        requirements = self._check_system_requirements()

        # Get GPU backend info
        gpu_backend = self.llama_installer._gpu_backend or "unknown"

        return {
            "llama": {
                "installed": llama_installed,
                "version": llama_version,
                "path": llama_path,
                "gpu_backend": gpu_backend,
            },
            "whisper": {
                "installed": whisper_installed,
                "version": whisper_version,
                "path": whisper_path,
            },
            "requirements": requirements,
            "bin_dir": str(self.bin_dir),
        }

    def _check_system_requirements(self) -> dict:
        """
        Check availability of build tools and system requirements.

        Returns:
            Dict with availability status of required tools
        """
        tools = {
            "gcc": bool(shutil.which("gcc")),
            "cmake": bool(shutil.which("cmake")),
            "git": bool(shutil.which("git")),
            "make": bool(shutil.which("make")),
        }

        logger.debug(f"System requirements: {tools}")
        return tools

    async def install_llama_from_release(
        self,
        backend: Optional[str] = None,
    ) -> bool:
        """
        Install llama.cpp from GitHub releases.

        Args:
            backend: GPU backend to use ('cuda', 'metal', 'cpu', or None for auto)

        Returns:
            True if installation succeeded, False otherwise
        """
        logger.info("Installing llama.cpp from release")

        try:
            if backend is None:
                backend = await self.llama_installer.detect_gpu_backend()

            success = await self.llama_installer.install(backend=backend)

            if success:
                version = await self.llama_installer.get_installed_version()
                logger.info(f"[success]Successfully installed llama.cpp {version}[/success]")
                return True
            else:
                logger.error("[error]Failed to install llama.cpp[/error]")
                return False
        except Exception as e:
            logger.error(f"[error]Error installing llama.cpp: {e}[/error]")
            return False

    async def install_llama_from_source(
        self,
        git_url: str = "https://github.com/ggerganov/llama.cpp.git",
        branch: str = "master",
    ) -> bool:
        """
        Install llama.cpp from source code.

        Args:
            git_url: Git repository URL
            branch: Git branch to build

        Returns:
            True if build succeeded, False otherwise
        """
        logger.info(f"Building llama.cpp from source (branch: {branch})")

        # Check requirements
        requirements = self._check_system_requirements()
        required = ["gcc", "cmake", "git"]
        missing = [tool for tool in required if not requirements[tool]]

        if missing:
            error_msg = f"Missing required tools: {', '.join(missing)}"
            logger.error(f"[error]{error_msg}[/error]")
            return False

        try:
            # Clone repository
            repo_dir = self.bin_dir / "llama-cpp-src"
            logger.info(f"[info]Cloning {git_url} to {repo_dir}[/info]")

            if repo_dir.exists():
                logger.debug("Repository already exists, pulling latest")
                import subprocess

                subprocess.run(
                    ["git", "-C", str(repo_dir), "pull", "origin", branch],
                    check=True,
                )
            else:
                import subprocess

                subprocess.run(
                    ["git", "clone", "-b", branch, git_url, str(repo_dir)],
                    check=True,
                )

            # Build
            logger.info("[info]Building llama.cpp[/info]")

            import subprocess

            # Detect GPU backend for build flags
            backend = await self.llama_installer.detect_gpu_backend()
            build_cmd = ["make", "-C", str(repo_dir)]

            if backend == "cuda":
                build_cmd.extend(["CUDA_PATH=/usr/local/cuda"])

            logger.debug(f"Build command: {' '.join(build_cmd)}")
            result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"[error]Build failed: {result.stderr}[/error]")
                return False

            # Copy binary
            binary_name = "llama-server"
            if self.llama_installer._platform == "windows":
                binary_name = "llama-server.exe"

            src_binary = repo_dir / binary_name
            if not src_binary.exists():
                src_binary = repo_dir / "build" / "bin" / binary_name

            if not src_binary.exists():
                logger.error(f"[error]Built binary not found at {src_binary}[/error]")
                return False

            dest_binary = self.bin_dir / binary_name
            logger.info(f"[info]Copying binary to {dest_binary}[/info]")
            shutil.copy2(src_binary, dest_binary)
            dest_binary.chmod(0o755)

            version = await self.llama_installer.get_installed_version()
            logger.info(f"[success]Successfully built llama.cpp {version}[/success]")
            return True

        except Exception as e:
            logger.error(f"[error]Error building llama.cpp: {e}[/error]")
            return False

    async def install_whisper_from_release(
        self,
        backend: Optional[str] = None,
    ) -> bool:
        """
        Install whisper.cpp from GitHub releases.

        Args:
            backend: GPU backend to use (optional, auto-detected)

        Returns:
            True if installation succeeded, False otherwise
        """
        logger.info("Installing whisper.cpp from release")

        try:
            if backend is None:
                backend = await self.whisper_installer.detect_gpu_backend()

            success = await self.whisper_installer.install(backend=backend)

            if success:
                version = await self.whisper_installer.get_installed_version()
                logger.info(f"[success]Successfully installed whisper.cpp {version}[/success]")
                return True
            else:
                logger.error("[error]Failed to install whisper.cpp[/error]")
                return False
        except Exception as e:
            logger.error(f"[error]Error installing whisper.cpp: {e}[/error]")
            return False

    async def install_whisper_from_source(
        self,
        git_url: str = "https://github.com/ggerganov/whisper.cpp.git",
        branch: str = "master",
    ) -> bool:
        """
        Install whisper.cpp from source code.

        Args:
            git_url: Git repository URL
            branch: Git branch to build

        Returns:
            True if build succeeded, False otherwise
        """
        logger.info(f"Building whisper.cpp from source (branch: {branch})")

        # Check requirements
        requirements = self._check_system_requirements()
        required = ["gcc", "cmake", "git"]
        missing = [tool for tool in required if not requirements[tool]]

        if missing:
            error_msg = f"Missing required tools: {', '.join(missing)}"
            logger.error(f"[error]{error_msg}[/error]")
            return False

        try:
            # Clone repository
            repo_dir = self.bin_dir / "whisper-cpp-src"
            logger.info(f"[info]Cloning {git_url} to {repo_dir}[/info]")

            if repo_dir.exists():
                logger.debug("Repository already exists, pulling latest")
                import subprocess

                subprocess.run(
                    ["git", "-C", str(repo_dir), "pull", "origin", branch],
                    check=True,
                )
            else:
                import subprocess

                subprocess.run(
                    ["git", "clone", "-b", branch, git_url, str(repo_dir)],
                    check=True,
                )

            # Build
            logger.info("[info]Building whisper.cpp[/info]")

            import subprocess

            build_cmd = ["make", "-C", str(repo_dir)]
            logger.debug(f"Build command: {' '.join(build_cmd)}")
            result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"[error]Build failed: {result.stderr}[/error]")
                return False

            # Copy binaries
            binaries = ["main", "stream", "server"]
            for binary_name in binaries:
                src_binary = repo_dir / binary_name
                if self.whisper_installer._platform == "windows":
                    src_binary = repo_dir / f"{binary_name}.exe"

                if src_binary.exists():
                    # Rename 'server' to 'whisper-server' for consistency
                    if binary_name == "server":
                        if self.whisper_installer._platform == "windows":
                            dest_binary = self.bin_dir / "whisper-server.exe"
                        else:
                            dest_binary = self.bin_dir / "whisper-server"
                    else:
                        dest_binary = self.bin_dir / src_binary.name
                    logger.debug(f"Copying {src_binary.name} to {dest_binary}")
                    shutil.copy2(src_binary, dest_binary)
                    dest_binary.chmod(0o755)

            version = await self.whisper_installer.get_installed_version()
            logger.info(f"[success]Successfully built whisper.cpp {version}[/success]")
            return True

        except Exception as e:
            logger.error(f"[error]Error building whisper.cpp: {e}[/error]")
            return False

    async def get_system_requirements(self) -> dict:
        """
        Get system requirements and available tools.

        Returns:
            Dict with tool availability and system info
        """
        logger.debug("Checking system requirements")

        requirements = self._check_system_requirements()

        # Get GPU backend info
        try:
            gpu_backend = await self.llama_installer.detect_gpu_backend()
        except Exception:
            gpu_backend = "unknown"

        return {
            "tools": requirements,
            "gpu_backend": gpu_backend,
            "bin_dir": str(self.bin_dir),
        }
