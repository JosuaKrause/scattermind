# Changelog

## [0.5.0] - 2024-06-27

### Added

- network drive friendly IO ([#15])
- loading env files ([#15])
- add session module ([#15])
- add id tensor conversion ([#15])
- add user id generation ([#15])
- add session id generation ([#15])
- add session storage and implementations ([#15])
- add shutdown hooks for faster shutdown on keyboard interrupt ([#15])

### Change

- let user control task id ([#15])
- make result fully serializable ([#15])
- store format in result ([#15])
- make task inputs serializable ([#15])
- overwrite batch size from node via `batch_size` ([#15])
- configurable max_task_retries ([#15])

### Breaking

- `locality` is now a method and not a staticmethod anymore ([#15])
- new required node method `session_field` ([#15])

### Bug-Fixes

- Various bug fixes and improvements ([#15])

## [0.4.1] - 2024-04-21

### Added

- Deferred task computation. If two tasks would cache for the same key only
  one of them is actually computed and the other uses the cached
  result. ([#12])

### Change

- Notify executors when tasks are available. (no spin-wait anymore) ([#12])
- Notify when results are available. (no spin-wait anymore) ([#12])
- Update redipy. ([#12])

## [0.4.0] - 2024-03-27

### Added

- More API functionality. ([#10])
- Graph based caching. ([#10])
- Proper handling of ghost tasks.
  (tasks that are in a queue but don't exist anymore) ([#10])
- Add healthcheck for workers. ([#10])
- Add external version info. ([#10])
- Add redis executor. ([#10])
- Add exit code for workers. ([#10])

### Change

- Update redipy. ([#10])

## [0.3.4] - 2024-03-06

### Change

- Update redipy. ([#8])

## [0.3.3] - 2024-02-28

### Added

- Add 'bool' node argument. ([#7])
- Add default value for node arguments. ([#7])
- Add scratchspace for ROA. ([#7])

### Breaking

- New mandatory argument to RAMAccess. ([#7])

### Change

- Changed License to Apache 2. ([#7])

## [0.3.2] - 2024-02-24

### Added

- Add override for worker system device.

## [0.3.1] - 2024-02-24

### Added

- Allow folders to load worker graphs.

## [0.3.0] - 2024-02-19

### Added

- Add dedicated node strategy. ([#5])
- Graph namespaces. ([#6])

## [0.2.0] - 2024-02-02

### Breaking

- Node strategy changed its arguments. ([#3])

### Added

- Add API frontend. ([#3])
- Full documentation. ([#3])
- Allow plugin executors to be locally parallel. ([#3])

### Bug-Fixes

- Remove daemon flag from thread executor. ([#3])

[#3]: https://github.com/JosuaKrause/scattermind/pull/3
[#5]: https://github.com/JosuaKrause/scattermind/pull/5
[#6]: https://github.com/JosuaKrause/scattermind/pull/6
[#7]: https://github.com/JosuaKrause/scattermind/pull/7
[#10]: https://github.com/JosuaKrause/scattermind/pull/10
[#12]: https://github.com/JosuaKrause/scattermind/pull/12
[#15]: https://github.com/JosuaKrause/scattermind/pull/15
