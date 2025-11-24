# IHC4BC
https://www.kaggle.com/datasets/akbarnejad1991/ihc4bc-compressed/data

Instruction pour utiliser l'agent ssh
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/cle
git pull
```

Instruction pour installer les dépendances dans un environnement virtuel Python sur le CCUB
```bash
module load python
python3 -m venv venv
source venv/bin/activate
pip3 install --prefix=/work/imvia/in156281/ihc4bc/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/ihc4bc/venv/lib/python3.9/site-packages:$PYTHONPATH
pip3 list
```

Instruction pour configurer les variables d'environnement avant l'exécution
```bash
# Configurer les chemins pour éviter les erreurs de permissions
export PYTHONPATH=/work/imvia/in156281/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
export WANDB_CACHE_DIR=/work/imvia/in156281/.cache/wandb
export WANDB_CONFIG_DIR=/work/imvia/in156281/.config/wandb

# Créer les répertoires si nécessaires
mkdir -p /work/imvia/in156281/.cache/matplotlib
mkdir -p /work/imvia/in156281/.cache/wandb
mkdir -p /work/imvia/in156281/.config/wandb

# Lancer le script d'entraînement
python3 training.py
```
Lancer un job sur le CCUB avec le script `train.sh`

```bash
qsub train.sh
```
Surveiller les jobs en cours d'exécution
```bash
qstat
# supprimer un job
qdel <job_id>
```