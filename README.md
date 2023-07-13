# Action_Recognitionar: CTR-GCN

## Estrazione dei frame e generazione dei dati

Per estrarre i frame e generare i dati, utilizzare extraction-generation.

Notebook: `ActionRecognition - Data Preparation.ipynb` [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dany-el92/Action_Recognition/blob/main/notebooks/ActionRecognition%20-%20Data%20Preparation.ipynb)

Istruzioni per l'esecuzione:

- Scaricare i file ntu120_xsub_train.pkl e ntu120_xsub_val.pkl, e posizionarli nella root del progetto extraction-generation;
- Modificare il file classes.json individuando le classi da utilizzare per ogni categoria di azione;
- Inserire nella cartella `videos` i video da cui estrarre i frame;
- Eseguire il comando `python gen_group.py` per generare il file groups.json, conterrà i gruppi di oggetti utilizzate nelle azioni;
- Eseguire il comando `python extract-frames.py` per estrarre i frame dai video;
- Eseguire il comando `python generator.py` per generare i dati da utilizzare per l'addestramento e il test dei modelli (output cartella `runs`).

Requisiti:

- MMPOSE: [https://github.com/open-mmlab/mmpose#installation](https://github.com/open-mmlab/mmpose#installation)
- ntu120_xsub_train.pkl: [https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_train.pkl](https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_train.pkl)
- ntu120_xsub_val.pkl: [https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_val.pkl](https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_val.pkl)

## Generazione dataset, training e test dei modelli

Per generare il dataset, addestrare e testare i modelli, utilizzare ctr-gcn-extended.

Notebook: `ActionRecognition - Training.ipynb` [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dany-el92/Action_Recognition/blob/main/notebooks/ActionRecognition%20-%20Training.ipynb)

Istruzioni per l'esecuzione della generazione dei file dataset:

- In `ctr-gcn-extended/data/ntu120` inserire la cartella `runs` generata da extraction-generation, e i file ntu120_xsub_train.pkl e ntu120_xsub_val.pkl;
- Eseguire il comando `python geb_statistics.py` per generare la cartella `statistics` contenente i file con le statistiche dei gruppi di oggetti;
- Eseguire il comando `python get_raw_skes_data.py`;
- Eseguire il comando `python get_raw_denoised_data.py`;
- Modificare `seq_transformation.py` per scegliere la combinazione di dati da utilizzare per l'addestramento e il test dei modelli: individuare le liste `train_ids` e `test_ids` e inserire gli id che si vorranno utilizzare;
- Eseguire il comando `python seq_transformation.py`, per ottenere i file `NTU120_CS.npz` e `NTU120_CV.npz`.

Istruzioni per l'esecuzione del training:

- Modificare il file config (es. `config/nturgbd120-cross-subject/default.yaml`) per scegliere i parametri del training. Specificare il file `.npz` da utilizzare per il training e il test;
- Eseguire il comando `python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/ctrgcn --device 0`;
- I risultati saranno contenuti nella work-dir specificata.

Istruzioni per l'esecuzione del test:

- Modificare il file config (es. `config/nturgbd120-cross-subject/default.yaml`) per scegliere i parametri del test. Specificare il file `.npz` da utilizzare per il test;
- Eseguire il comando `python main.py --config work_dir/ntu120/csub/ctrgcn/config.yaml --work-dir work_dir/ntu120/csub/ctrgcn --phase test --save-score True --weights work_dir/ntu120/csub/ctrgcn/runs-100-16300.pt --device 0`. Modificare il parametro `--weights` con il path del file `.pt` generato dal training;

Requisiti:

- [https://github.com/Uason-Chen/CTR-GCN/#prerequisites](https://github.com/Uason-Chen/CTR-GCN/#prerequisites)
- [https://github.com/Uason-Chen/CTR-GCN/#data-preparation](https://github.com/Uason-Chen/CTR-GCN/#data-preparation)
- eseguire `pip install -r requirements.txt`
- ntu120_xsub_train.pkl: [https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_train.pkl](https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_train.pkl)
- ntu120_xsub_val.pkl: [https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_val.pkl](https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_val.pkl)

## Webapp

Il modello dovrà essere posizionato nella cartella `webapp/ar-ctrgcn-webapp/models`. Sarà necessario anche modificare il file `ar-ctrgcn-app.py` per indicare il path e il nome del modello da utilizzare: per far ciò, modificare la variabile `workdir`, `weights` e `model_config`.

Per l'avvio della webapp, eseguire dalla cartella `webapp` il dockerfile con i seguenti comandi:

- `cd webapp`
- `docker build -t ctr-gcn-webapp .`
- `docker run --publish 5000:5000 ctr-gcn-webapp`

L'esecuzione della webapp sarà disponibile all'indirizzo [http://localhost:5000](http://localhost:5000)

Requisiti:

- docker
