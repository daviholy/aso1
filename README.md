# Hledání anomálií v signálech ICP, ABP

## požadavky

- python3.11
- [poetry](https://github.com/python-poetry/poetry)
- [charis](https://physionet.org/content/charisdb/1.0.0/) dataset
  
## Postup pro zprovoznění

### Dataset

Překopirovat do složky kam je potřeba, program defaultně očekává ve sloźce data v kořeni repositáře.

### Program
1)     poetry install
    
2)     poetry shell
3)     python -m src ...

## Návod k obsluze

seznamy příkazů a jejich obsluha lze najit v příslušných pomocných obrazovkách.

př

- seznam příkazů: ```python -m src --help```
- popis příkazu graph ```python -m src graph --help```
