# Script per Organizzare Dataset

Questo script Python organizza i dataset da file CSV nella struttura richiesta.

## Funzionalità

- **Lettura automatica** di tutti i file CSV dalla cartella `data_csv/`
- **Estrazione automatica** del nome dataset e split dal percorso del file
- **Filtro automatico** per righe con `mask_path` non vuoto (quando presente nell'header)
- **Organizzazione** nella struttura: `base_path/dataset/split/classe/`
- **Logging completo** delle operazioni
- **Modalità dry-run** per testare senza copiare file

## Utilizzo

### Esempi di base

```bash
# Processa tutti i CSV e organizza i file
python organize_dataset.py --base_path /path/to/organized_data --source_base_path /path/to/source_images

# Processa solo un file CSV specifico
python organize_dataset.py --base_path ./organized_data --csv_file data_csv/HC18/hc18_val.csv

# Modalità dry-run (simulazione senza copiare file)
python organize_dataset.py --base_path ./test --dry_run
```

### Parametri

- `--base_path` / `-b`: **[RICHIESTO]** Percorso base per l'output
- `--source_base_path` / `-s`: Percorso base per i file sorgente (opzionale)
- `--csv_file` / `-c`: File CSV specifico da processare (opzionale)
- `--data_csv_dir` / `-d`: Directory contenente i CSV (default: `data_csv`)
- `--dry_run`: Esegue simulazione senza copiare file

## Esempio con HC18

Per il file `data_csv/HC18/hc18_val.csv`:

```bash
python organize_dataset.py --base_path ./organized_data --source_base_path /path/to/hc18_images
```

Questo creerà la struttura:
```
organized_data/
└── HC18/
    └── val/
        └── Head/
            ├── 645_HC.png
            ├── 405_HC.png
            └── ...
```

## Comportamento del Filtro

- **Con colonna `mask_path`**: Include solo righe con `mask_path` non vuoto
- **Senza colonna `mask_path`**: Include tutte le righe

## Log

Lo script genera:
- **Log su console**: Informazioni in tempo reale
- **File di log**: `organize_dataset.log` con dettagli completi

## Struttura Output

```
base_path/
├── DATASET1/
│   ├── train/
│   │   ├── classe1/
│   │   └── classe2/
│   ├── val/
│   │   ├── classe1/
│   │   └── classe2/
│   └── test/
│       ├── classe1/
│       └── classe2/
└── DATASET2/
    └── ...
```

## Estrazione Automatica

- **Dataset**: Estratto dal nome della cartella (es. `HC18` da `data_csv/HC18/`)
- **Split**: Estratto dal nome del file (es. `val` da `hc18_val.csv`)
- **Classe**: Letta dalla colonna `class` del CSV

## Gestione Errori

Lo script gestisce automaticamente:
- File sorgente mancanti (warning nel log)
- Righe CSV con dati mancanti (saltate)
- Errori di copia file (logged)
- Directory mancanti (create automaticamente)
