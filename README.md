# Simple Bot for StarCraft2

- [ ] MoveToBeacon

- [x] BuildMarines

- [x] CollectiMineralsAndGas

- [x] CollectMineraShards

- [ ] DefeatZerglingsAndBanelings

- [ ] FindAndDefeatZerglings

- [ ] DefeatRoaches

## Dependency
- pysc2
- [mini games](https://github.com/deepmind/pysc2)

## Test Example
```python
python -m pysc2.bin.agent --map <map name> --agent <agent module>
```
For example 
```python
python -m pysc2.bin.agent --map BuildMarines --agent BuildMarines.BuildMarines
```