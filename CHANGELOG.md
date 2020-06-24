# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.0]
### Added

- Source code from PyGran repo
- CHANGELOG for version logging
- CircleCI config for continuous integration
- Added License file

## [1.1.0]
### Added
- Submodule 'generator' from PyGran.simulation
- Support for VTK file formats: vtp + vtu

### Changed
- Method computeRDF() in core.py*: added optional 'npts' arg
- Function hcp() in generator.py: added optional 'units' arg

### Removed
- Method computeGCOM() in core.py*
- Attribute 'length' for Particles.molecule

## [1.1.1]
### Changed
- Fixed bug with computeIntensitySegregation: 'self.types' typo
