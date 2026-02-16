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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cyber_inference.core.config import get_settings
from cyber_inference.core.database import get_db_session
from cyber_inference.core.logging import get_logger
from cyber_inference.models.db_models import Configuration
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

    async def _save_binary_path(self, binary_name: str, path: Path) -> None:
        """Save binary installation path to database."""
        key = f"binary_path_{binary_name}"
        try:
            async with get_db_session() as session:
                # Check if exists
                stmt = select(Configuration).where(Configuration.key == key)
                result = await session.execute(stmt)
                config = result.scalar_one_or_none()
                
                if config:
                    config.value = str(path)
                else:
                    config = Configuration(
                        key=key,
                        value=str(path),
                        value_type="string",
                        description=f"Installed path for {binary_name} binary"
                    )
                    session.add(config)
                
                await session.commit()
                logger.debug(f"Saved {binary_name} path to database: {path}")
        except Exception as e:
            logger.warning(f"Failed to save {binary_name} path to database: {e}")

    async def _get_saved_binary_path(self, binary_name: str) -> Optional[Path]:
        """Get saved binary installation path from database."""
        key = f"binary_path_{binary_name}"
        try:
            async with get_db_session() as session:
                stmt = select(Configuration).where(Configuration.key == key)
                result = await session.execute(stmt)
                config = result.scalar_one_or_none()
                
                if config and config.value:
                    path = Path(config.value)
                    if path.exists():
                        logger.debug(f"Found saved {binary_name} path: {path}")
                        return path
                    else:
                        logger.debug(f"Saved {binary_name} path no longer exists: {path}")
                        return None
        except Exception as e:
            logger.debug(f"Failed to load {binary_name} path from database: {e}")
        
        return None

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
                # Save binary path to database
                binary_path = self.llama_installer.get_binary_path()
                await self._save_binary_path("llama", binary_path)
                
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

            # Build using CMake
            logger.info("[info]Building llama.cpp with CMake[/info]")

            import subprocess

            build_dir = repo_dir / "build"
            
            # Detect GPU backend for CMake flags
            backend = await self.llama_installer.detect_gpu_backend()
            cmake_flags = []
            
            if backend == "cuda":
                cmake_flags.append("-DGGML_CUDA=ON")
            elif backend == "metal":
                cmake_flags.append("-DGGML_METAL=ON")
            
            # Run cmake to configure build
            cmake_cmd = ["cmake", "-B", str(build_dir)] + cmake_flags + [str(repo_dir)]
            logger.debug(f"CMake command: {' '.join(cmake_cmd)}")
            result = subprocess.run(cmake_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"[error]CMake configuration failed:[/error]")
                logger.error(result.stderr)
                return False
            
            # Run cmake build
            build_invoke_cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]
            logger.debug(f"Build command: {' '.join(build_invoke_cmd)}")
            result = subprocess.run(build_invoke_cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"[error]Build failed:[/error]")
                logger.error(result.stderr)
                return False

            # Copy binary - llama.cpp builds llama-server in build/bin/
            binary_name = "llama-server"
            if self.llama_installer._platform == "windows":
                binary_name = "llama-server.exe"
            
            src_binary = build_dir / "bin" / binary_name
            if not src_binary.exists():
                # Fallback to build directory root
                src_binary = build_dir / binary_name
            
            if not src_binary.exists():
                # Last fallback to repo root
                src_binary = repo_dir / binary_name
            
            if not src_binary.exists():
                logger.error(f"[error]Built binary not found - checked:[/error]")
                logger.error(f"  {build_dir / 'bin' / binary_name}")
                logger.error(f"  {build_dir / binary_name}")
                logger.error(f"  {repo_dir / binary_name}")
                return False

            dest_binary = self.bin_dir / binary_name
            logger.info(f"[info]Copying binary from {src_binary} to {dest_binary}[/info]")
            shutil.copy2(src_binary, dest_binary)
            dest_binary.chmod(0o755)

            # Save binary path to database
            binary_path = self.llama_installer.get_binary_path()
            await self._save_binary_path("llama", binary_path)
            
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
                # Save binary path to database
                binary_path = self.whisper_installer.get_binary_path()
                await self._save_binary_path("whisper", binary_path)
                
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

            # Build using CMake
            logger.info("[info]Building whisper.cpp with CMake[/info]")

            import subprocess

            build_dir = repo_dir / "build"
            
            # Run cmake to configure build
            cmake_cmd = ["cmake", "-B", str(build_dir), str(repo_dir)]
            logger.debug(f"CMake command: {' '.join(cmake_cmd)}")
            result = subprocess.run(cmake_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"[error]CMake configuration failed:[/error]")
                logger.error(result.stderr)
                return False
            
            # Run make to build
            make_cmd = ["make", "-C", str(build_dir)]
            logger.debug(f"Make command: {' '.join(make_cmd)}")
            result = subprocess.run(make_cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"[error]Build failed:[/error]")
                logger.error(result.stderr)
                return False

            # Copy binaries - whisper.cpp builds them in build/bin/ or root
            binaries_to_copy = ["whisper-server", "main", "stream"]
            copied_count = 0
            
            for binary_name in binaries_to_copy:
                src_binary = None
                
                # Check build/bin/ directory first (CMake standard)
                build_bin_path = build_dir / "bin" / binary_name
                if build_bin_path.exists():
                    src_binary = build_bin_path
                # Fallback to build/ directory
                elif (build_dir / binary_name).exists():
                    src_binary = build_dir / binary_name
                # Fallback to repo root
                elif (repo_dir / binary_name).exists():
                    src_binary = repo_dir / binary_name
                
                # Check for Windows .exe extension
                if src_binary is None and self.whisper_installer._platform == "windows":
                    exe_name = f"{binary_name}.exe"
                    build_bin_path = build_dir / "bin" / exe_name
                    if build_bin_path.exists():
                        src_binary = build_bin_path
                    elif (build_dir / exe_name).exists():
                        src_binary = build_dir / exe_name
                    elif (repo_dir / exe_name).exists():
                        src_binary = repo_dir / exe_name
                
                if src_binary:
                    dest_binary = self.bin_dir / src_binary.name
                    logger.info(f"[info]Copying {src_binary.name} to {dest_binary}[/info]")
                    shutil.copy2(src_binary, dest_binary)
                    dest_binary.chmod(0o755)
                    copied_count += 1
                else:
                    logger.warning(f"[warning]Binary not found: {binary_name}[/warning]")
            
            # Verify whisper-server was copied (required)
            whisper_server_path = self.bin_dir / "whisper-server"
            if self.whisper_installer._platform == "windows":
                whisper_server_path = self.bin_dir / "whisper-server.exe"
            
            if not whisper_server_path.exists():
                logger.error("[error]whisper-server binary not found after build[/error]")
                logger.error(f"[error]Expected at: {whisper_server_path}[/error]")
                logger.error(f"[error]Found {copied_count} binaries in {self.bin_dir}[/error]")
                return False

            # Save binary path to database
            binary_path = self.whisper_installer.get_binary_path()
            await self._save_binary_path("whisper", binary_path)
            
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
