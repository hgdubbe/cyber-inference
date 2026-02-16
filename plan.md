# Implementation Plan: Custom Chat Templates, Custom Bin Directory, and Binary Installers UI

## Overview
This plan outlines the implementation of three interconnected features for Cyber-Inference:
1. **Custom Jinja2 Chat Template Loading** - Allow users to provide custom prompt templates for formatting model conversations
2. **Custom Bin Directory Setting** - Allow configuration of where binaries (llama.cpp, whisper.cpp) are stored
3. **Binary Installer UI Buttons** - Add web UI buttons to install/reinstall llama.cpp and whisper.cpp from releases or source

---

## Topic 1: Custom Jinja2 Chat Template Loading

### Goals
- Allow users to provide custom Jinja2 chat/prompt templates for language models
- Support different chat templates for different models (system prompts, message formatting)
- Maintain backward compatibility with default templates
- Enable chat template directory configuration via environment variables
- Use custom templates in model inference (chat completions endpoints)

### Current State
- Chat templates are not currently implemented in the codebase
- Model inference is handled in API (`src/cyber_inference/api/v1.py`)
- No template management system for model prompts exists
- Models likely send raw prompts directly to llama.cpp

### Implementation Strategy

#### 1.1 Configuration Extension
**File:** `src/cyber_inference/core/config.py`
- Add new configuration fields:
  - `chat_templates_dir`: Optional[Path] = None (env: `CYBER_INFERENCE_CHAT_TEMPLATES_DIR`)
  - Default to `None` (no custom templates, use basic prompt formatting)
  - Validate directory exists and is readable

#### 1.2 Chat Template Manager Service
**File:** `src/cyber_inference/services/chat_template_manager.py` (NEW)
- Create new service class `ChatTemplateManager`
- Methods:
  - `load_template(model_name: str) -> str` - Load chat template for specific model
  - `get_available_templates() -> list[str]` - List available template files
  - `render_chat_template(template: str, messages: list, system_prompt: Optional[str]) -> str` - Render template with Jinja2
  - `validate_templates() -> dict` - Check if custom templates are valid Jinja2
  - `get_template_path(name: str) -> Optional[Path]` - Get path to template file

- Template format:
  - Files named `{model_name}.jinja2` or `default.jinja2`
  - Jinja2 variables available: `messages`, `system_prompt`, `tools` (if applicable)
  - Messages format: list of `{"role": "user"|"assistant"|"system", "content": str}`

#### 1.3 Main Application Update
**File:** `src/cyber_inference/main.py`
- Initialize `ChatTemplateManager` in lifespan startup
- Log available chat templates

#### 1.4 API Integration
**File:** `src/cyber_inference/api/v1.py`
- Modify chat completion endpoint to:
  - Check if model has selected template (from request or database preference)
  - Load appropriate chat template for the model
  - Render user messages through template
  - Send formatted prompt to llama.cpp
  - Fallback to simple concatenation if no template available
  - Support template selection via API request parameter (`template` or `chat_template`)
  - Support saving template preference per model in database

#### 1.5 Admin API Endpoints
**File:** `src/cyber_inference/api/admin.py`
- Add endpoints:
  - `GET /admin/chat-templates` - List available chat templates
  - `GET /admin/chat-templates/{model_name}` - Get template for specific model
  - `POST /admin/validate-chat-templates` - Validate custom template directory
  - `POST /admin/chat-templates/preview` - Preview rendered template with sample messages

#### 1.6 Dashboard UI Integration
**File:** Web templates (dashboard.html)
- Add chat template dropdown selector to dashboard showing:
  - List of available templates
  - Current selected template for active model
  - Dropdown to switch templates on-the-fly
  - Visual indicator of which template is being used
  - Brief description of each template
- Template selection updates model's chat template preference
- Selection persists across requests (stored in database or session)

#### 1.7 Database Model Extension
**File:** `src/cyber_inference/models/db_models.py`
- Update `ModelSession` or `Model` to store selected chat template:
  - `selected_chat_template`: Optional[str] - Which template this model uses
  - Default to `None` (uses model default or system default)

### Validation & Testing
- Create sample chat templates for popular models (llama2, mistral, etc.)
- Verify Jinja2 template rendering with different message formats
- Test fallback to default behavior when no template available
- Ensure system prompts are properly injected
- Test with missing custom directory (should use defaults)
- Test with invalid Jinja2 syntax in custom templates (show helpful error)
- Test dashboard dropdown loads templates correctly
- Test switching templates on-the-fly updates API calls
- Test template selection persists across page reloads

---

## Topic 2: Custom Bin Directory Setting

### Goals
- Make `bin_dir` fully configurable at runtime
- Allow users to share bin directory across multiple installations
- Enable database persistence of bin directory choice
- Provide UI to change bin directory

### Current State
- `bin_dir` is configured in `Settings` with default `Path.cwd() / "bin"`
- Already supports `CYBER_INFERENCE_BIN_DIR` environment variable
- Used by `ProcessManager` and both installer classes
- Not currently changeable via admin UI

### Implementation Strategy

#### 2.1 Database Model Update
**File:** `src/cyber_inference/models/db_models.py`
- Add to `Configuration` table if not already tracked:
  - Already exists in config, just needs to be exposed

#### 2.2 Config Update
**File:** `src/cyber_inference/core/config.py`
- `bin_dir` already exists as a setting
- Add validation:
  - Ensure directory is accessible and writable
  - Validate path doesn't contain invalid characters
  - Check for sufficient disk space for binaries (~2GB)

#### 2.3 Runtime Configuration
**File:** `src/cyber_inference/core/config.py`
- Add `bin_dir` to `CONFIG_DB_CASTS` dictionary
- Implement dynamic reload mechanism for bin directory changes

#### 2.4 Admin API Endpoints
**File:** `src/cyber_inference/api/admin.py`
- Add endpoints:
  - `PUT /admin/config/bin_dir` - Update binary directory
  - `POST /admin/validate-bin-dir` - Validate directory before committing
  - Include validation response with disk space, permissions check
  - Return current bin directory with available space in `GET /admin/config`

#### 2.5 Web UI Updates
**File:** Template updates needed
- Add bin directory display and change form in settings page
- Show:
  - Current bin directory path
  - Available binaries in current directory
  - Disk space available
  - Warning if not writable or insufficient space

#### 2.6 ProcessManager Update
**File:** `src/cyber_inference/services/process_manager.py`
- Ensure `bin_dir` is passed consistently during initialization
- Support runtime updates to `bin_dir` (reload process manager if needed)

### Validation & Testing
- Verify environment variable takes precedence
- Test database persistence of bin directory
- Confirm ProcessManager uses updated bin directory
- Test with read-only directories (should fail gracefully)
- Verify installers use the configured bin directory

---

## Topic 3: Binary Installer UI and CLI

### Goals
- Provide UI buttons to install/reinstall llama.cpp and whisper.cpp
- Support installation from source code (git clone + make)
- Support installation from GitHub releases (precompiled binaries)
- Show installation progress in real-time via WebSocket
- Display current binary versions and status

### Current State
- `LlamaInstaller` class exists and can download/install binaries
- `WhisperInstaller` class exists and can download/install binaries
- Both support platform detection and GPU backend detection
- Already have methods: `install()`, `get_latest_release()`, `detect_gpu_backend()`
- WebSocket mechanism exists for real-time log streaming

### Implementation Strategy

#### 3.1 Enhanced Installer Services
**File:** `src/cyber_inference/services/llama_installer.py` & `whisper_installer.py`

**Add to LlamaInstaller:**
- `async build_from_source(git_url: str, branch: str = "master") -> bool`
  - Clone repository
  - Detect build requirements (cmake, gcc, make)
  - Run make with detected GPU backend
  - Move compiled binary to bin_dir
  
- `async get_installed_version() -> Optional[str]`
  - Run `llama-server --version` to detect installed version
  
- `async get_installation_status() -> dict`
  - Check if binary exists
  - Get version if available
  - Check GPU backend support

**Add to WhisperInstaller:**
- `async build_from_source(git_url: str, branch: str = "master") -> bool`
- `async get_installed_version() -> Optional[str]`
- `async get_installation_status() -> dict`

#### 3.2 Installation Manager Service
**File:** `src/cyber_inference/services/installation_manager.py` (NEW)
- Create `InstallationManager` class
- Manages both llama.cpp and whisper.cpp installations
- Methods:
  - `async install_llama_from_release(backend: Optional[str] = None) -> bool`
  - `async install_llama_from_source(branch: str = "master") -> bool`
  - `async install_whisper_from_release(backend: Optional[str] = None) -> bool`
  - `async install_whisper_from_source(branch: str = "master") -> bool`
  - `async get_system_requirements() -> dict` - Check cmake, gcc, make availability
  - `async get_installation_status() -> dict` - Status for both binaries

- Features:
  - Progress tracking via logging at INFO level
  - Proper error handling and cleanup on failure
  - Timeout handling for long-running operations
  - Write progress to dedicated log channel for WebSocket streaming

#### 3.3 Admin API Endpoints
**File:** `src/cyber_inference/api/admin.py`
- Add endpoints:
  - `GET /admin/binaries/status` - Get installation status for both binaries
  - `POST /admin/binaries/install` - Install from release
    ```json
    { 
      "binary": "llama" | "whisper",
      "source": "release" | "source",
      "branch": "master" (for source builds)
    }
    ```
  - `POST /admin/binaries/check-requirements` - Check build requirements
  - `GET /admin/binaries/versions` - Get available and installed versions
  - WebSocket streaming of progress via existing WebSocket handler
  - Return installation progress and status in real-time

#### 3.4 Web UI Updates
**Files:** Template updates needed
- Dashboard/settings page additions:
  - New "Binaries" or "Tools" section showing:
    - Current installation status of llama.cpp
    - Current installation status of whisper.cpp
    - Installed version numbers (if available)
    - GPU backend detection result
    - Available disk space
    - System requirements check

  - Installation buttons:
    - "Install from Release" button for each binary
    - "Install from Source" button for each binary (with branch selector)
    - "Reinstall" button for each binary
    - "Check Requirements" button to verify build dependencies

  - Real-time progress display:
    - Progress modal/panel showing installation logs
    - Live log streaming via WebSocket
    - Cancel button to abort installation (if possible)
    - Spinner/progress bar during installation

- Settings UI:
  - Show current bin directory
  - Show available space for binaries
  - Disk space warnings if < 5GB available

#### 3.5 Configuration Models
**File:** `src/cyber_inference/models/schemas.py`
- Add new Pydantic models:
  ```python
  class BinaryStatus(BaseModel):
      installed: bool
      version: Optional[str]
      path: Optional[Path]
      gpu_backend: str
      size_mb: Optional[float]

  class SystemRequirements(BaseModel):
      gcc: bool
      cmake: bool
      git: bool
      make: bool
      required_disk_space_gb: int
      available_disk_space_gb: int

  class InstallationStatus(BaseModel):
      llama: BinaryStatus
      whisper: BinaryStatus
      requirements: SystemRequirements
      bin_dir: Path
      available_space_gb: float

  class InstallBinaryRequest(BaseModel):
      binary: Literal["llama", "whisper"]
      source: Literal["release", "source"]
      branch: str = "master"
  ```

#### 3.6 CLI Command Extension
**File:** `src/cyber_inference/cli.py`
- Add/enhance commands:
  - `cyber-inference install-llama [--from-source] [--branch BRANCH]`
  - `cyber-inference install-whisper [--from-source] [--branch BRANCH]`
  - `cyber-inference binary-status` - Show status of both binaries

### Validation & Testing
- Test installation from release (check binary exists and runs)
- Test installation from source (build and verify)
- Test progress reporting via WebSocket
- Verify UI buttons correctly trigger installations
- Test with missing build tools (should show helpful error)
- Test installation cancellation
- Verify version detection works
- Test with different GPU backends
- Test disk space warnings

---

## Implementation Order and Dependencies

### Phase 1: Configuration & Foundation (Days 1-2)
1. Update `config.py` with `chat_templates_dir` setting
2. Create `chat_template_manager.py` service
3. Create `installation_manager.py` service (for binaries)
4. Create sample chat templates for common models

### Phase 2: Admin API (Days 2-3)
1. Add chat template endpoints to `admin.py`
2. Add binary installation endpoints to `admin.py`
3. Add config update endpoints for bin_dir
4. **Blocking:** Requires Phase 1 completion

### Phase 3: API Integration (Days 3-4)
1. Update inference endpoint in `v1.py` to use chat templates
2. Integrate ChatTemplateManager with model inference
3. Test template rendering in inference

### Phase 4: Web UI (Days 4-5)
1. Update settings template with bin_dir config
2. Create/update binaries management template
3. Add real-time progress display
4. Add chat template dropdown to dashboard
5. Add chat templates section showing current templates
6. **Blocking:** Requires Phase 2 completion

### Phase 5: Enhanced Installers (Days 1-4, parallel)
1. Add version detection to both installers
2. Add status checking methods
3. Add from-source build methods
4. Test installers in isolation
5. **Blocking:** Can be mostly parallel with other phases

### Phase 6: Integration & Testing (Days 5-7)
1. Integration testing of all components
2. UI/UX testing
3. Edge case handling
4. Documentation updates
5. **Blocking:** Requires all previous phases

---

## File Modifications Summary

### New Files to Create
- `src/cyber_inference/services/chat_template_manager.py`
- `src/cyber_inference/services/installation_manager.py`
- Sample chat templates in `chat_templates/` directory:
  - `chat_templates/default.jinja2` (fallback template)
  - `chat_templates/llama2.jinja2`
  - `chat_templates/mistral.jinja2`
  - `chat_templates/openchat.jinja2`

### Files to Modify
1. `src/cyber_inference/core/config.py`
   - Add `chat_templates_dir` setting
   - Add validation for templates directory

2. `src/cyber_inference/models/db_models.py`
   - Update `ModelSession` or `Model` to include `selected_chat_template` field
   - Add column to store user's template preference per model

3. `src/cyber_inference/api/v1.py`
   - Import ChatTemplateManager
   - Update chat completion endpoint to:
     - Accept template parameter in request
     - Load selected template (from request, database, or default)
     - Render templates and send to llama.cpp
     - Save template preference if provided

4. `src/cyber_inference/api/admin.py`
   - Add chat template endpoints (list, get, validate, preview)
   - Add endpoint to set default template for a model
   - Add binary install endpoints
   - Add bin_dir update endpoints

5. `src/cyber_inference/models/schemas.py`
   - Add ChatTemplate schema (name, path, description, example)
   - Add ChatCompletionRequest update to include `template` parameter
   - Add BinaryStatus, SystemRequirements, InstallationStatus schemas
   - Add InstallBinaryRequest schema

6. `src/cyber_inference/main.py`
   - Initialize ChatTemplateManager in lifespan
   - Initialize InstallationManager in lifespan
   - Load available templates on startup

7. `src/cyber_inference/services/llama_installer.py`
   - Add version detection method
   - Add status checking method
   - Add from-source build method

8. `src/cyber_inference/services/whisper_installer.py`
   - Add version detection method
   - Add status checking method
   - Add from-source build method

9. `src/cyber_inference/cli.py`
   - Add/enhance install commands with --from-source option

10. Web Templates (updates)
    - Update `dashboard.html` with chat template dropdown selector in model/settings area
    - Update `settings.html` with:
      - Chat templates management section
      - Default template selection per model
      - Custom templates directory path display
      - Bin directory configuration
    - Update/create `binaries.html` with install UI
    - Create template partial `_chat_template_selector.html` for dropdown reuse
### Chat Template Strategy
- Jinja2 templates for formatting model prompts (not web UI)
- Variables available in templates:
  - `messages`: List of message dicts with `role` and `content`
  - `system_prompt`: System prompt string (optional)
  - `tools`: (Future extensibility for function calling)
- Template naming: `{model_name}.jinja2` or `default.jinja2` for fallback
- Load templates on startup and cache them in memory
- **Dashboard Dropdown Selector**:
  - Displays all available templates
  - Shows currently selected template for active model
  - Allows switching templates on-the-fly
  - Updates via AJAX request without page reload
  - Saves selection to database for persistence
- Render templates before sending to llama.cpp inference
- Log template used for debugging
- Support template override via API request parameter
- Save template selection per model for future sessions

### Chat Template Examples
```jinja2
{# default.jinja2 - Simple concatenation #}
{% for msg in messages %}
{{ msg.role }}: {{ msg.content }}

{% endfor %}
```

```jinja2
{# llama2.jinja2 - Llama 2 chat format #}
{% if system_prompt %}[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>
{% endif %}
{% for msg in messages %}
{% if msg.role == "user" %}
{{ msg.content }} [/INST]
{% elif msg.role == "assistant" %}
{{ msg.content }} </s><s>[INST]
{% endif %}
{% endfor %}
```

### Binary Installation Strategy
- Run installations in separate thread pool to avoid blocking
- Stream progress via WebSocket for UI updates
- Implement rollback in case of failed installation
- Maintain checksum verification for downloaded binaries
- Cleanup partial downloads on failure

### Configuration Persistence
- Use existing database configuration system
- Support environment variable override
- Log configuration changes with timestamps
- Provide config rollback mechanism

### Error Handling
- Graceful template rendering errors (show error but continue)
- Helpful error messages for disk space issues
- Build requirement checking before attempting source builds
- Process cleanup on installation failure
- Invalid template syntax should be caught and logged

### Performance
- Cache compiled Jinja2 templates in memory
- Lazy load templates only when model is loaded
- Parallelize version detection for both binaries
- Use async operations for all network operations

---

## Testing Strategy

### Unit Tests
- Chat template rendering with different message formats
- Template validation (Jinja2 syntax checking)
- Template path resolution
- Configuration validation
- Installer version detection
- Installation status checking

### Integration Tests
- End-to-end chat completion with custom template
- Fallback to default template when custom missing
- Template rendering in actual inference calls
- End-to-end installation from release
- End-to-end installation from source
- Configuration persistence and reload

### UI Tests
- Binary installation button click handlers
- Form submission and validation
- Progress display updates
- Real-time log streaming
- Chat template dropdown selector:
  - Dropdown loading and rendering
  - Template switching via dropdown
  - AJAX update without page reload
  - Persistence across page reloads
  - Correct template application in API calls
- Chat templates display in settings

### System Tests
- Full installation workflow
- Custom chat templates with different models
- Chat template dropdown selector:
  - Switch templates on active model
  - Verify inference uses correct template
  - Test persistence across sessions
  - Test with multiple models (each has own preference)
- Configuration changes reflected in running app
- Multiple binary installations concurrently
- Template rendering does not break inference

---

## Documentation Updates

1. Update README.md with:
   - Custom chat template loading documentation
   - Custom bin directory configuration
   - Installation methods (UI vs CLI)
   - Chat template format and examples

2. Update AGENTS.md with:
   - New services and their roles
   - New environment variables
   - Chat template structure
   - New error handling patterns

3. Create INSTALLATION-GUIDE.md with:
   - Step-by-step installation from release
   - Step-by-step installation from source
   - Troubleshooting guide
   - GPU backend configuration

4. Create CHAT-TEMPLATES-GUIDE.md with:
   - Chat template format and variables
   - Examples for popular models (llama2, mistral, etc.)
   - How to create custom templates
   - Template validation and debugging
   - Using the dashboard dropdown selector
   - Setting default templates per model
