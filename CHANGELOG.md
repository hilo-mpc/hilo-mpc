## v2.1.0 - OPC UA Release
### Added 
- Two boilerplates for OPC UA connectivity. One simple and with minimum setup `OPCUASimpleControlLoop` and the other thought to have more flexibility `OPCUAConnector`
- Tests for OPC UA Code

## v2.0.1 - Patch Release
### Added
- Documentation on observer module 
- Missing docstrings in the mhe and mpc methods

### Changed
- **pandas is now a required dependency** - Moved from optional to core dependencies as it's used in neural network implementation

## v2.0.0 - Major Release
### 💥 BREAKING CHANGES
- **Dropped Python 3.9 support** - now requires Python >=3.10
- Pandas version updated to >2.0.0
- Minimum Python version increased due to security dependencies

### 🔒 Security Fixes
- **Fixed Keras vulnerability CVE-2024-XXXX** - Added Keras >=3.11.3 requirement
- Addresses safe_mode bypass vulnerability in .h5/.hdf5 model loading

### 🏗️ Infrastructure Updates
- **Migrated to Poetry** for dependency management and packaging
- Updated CI/CD to test Python 3.10-3.12 only
- **Documentation workflow** now deploys only on version tags
- Updated GitHub Actions for automated PyPI publishing with Trusted Publishing

### 📚 Documentation
- **Comprehensive installation guide** with Poetry and pip instructions
- **Auto-version detection** in Sphinx documentation
- **Auto-updating copyright** year in documentation
- Updated contributor guidelines with simplified GitHub Flow

### 🧪 Testing & Quality
- All 515 tests passing with updated dependencies
- Enhanced security scanning with CodeQL
- Dependabot integration for automated security updates

## v1.2.0
- Extended compatibility to Casadi 3.7.2
- Extended compatibility up to Python 3.12
## v1.1.0
- Extended compatibility to CasADi 3.6.3
- Extended compatibility to Numpy 1.25.2
- Updated Bokeh maximum version in README.md 
- Added CHANGELOG