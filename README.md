# Ia2hl


# Instructions pour le CCUB
## Alias 'pull'
```bash
alias pull='eval "$(ssh-agent -s)" && ssh-add ~/.ssh/cle && git pull'
```
## Alias 'gpu'
```bash
alias gpu='qlogin -q gpu'
```
## Alias 'st'
```bash
git status
```
## Création de l'environnement virtuel
```bash
module load python
python3 -m venv venv
source venv/bin/activate
pip3 install --prefix=/work/imvia/in156281/ihc4bc/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/ihc4bc/venv/lib/python3.9/site-packages:$PYTHONPATH
pip3 list
```
## Alias 'venv'
```bash
mkdir -p /work/imvia/in156281/.cache/matplotlib
mkdir -p /work/imvia/in156281/.cache/wandb
mkdir -p /work/imvia/in156281/.config/wandb
alias venv='module load python && source venv/bin/activate 
                               && export PYTHONPATH=venv/lib/python3.9/site-packages:$PYTHONPATH
                               && export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
                               && export WANDB_CACHE_DIR=/work/imvia/in156281/.cache/wandb 
                               && export WANDB_CONFIG_DIR=/work/imvia/in156281/.config/wandb'
```

# Lancer le script d'entraînement
```bash
python3 training.py
```
Lancer un job sur le CCUB avec le script `train.sh`
```bash
qsub train.sh
```
Surveiller les jobs en cours d'exécution
```bash
qstat
```
Supprimer un job
```bash
qdel <job_id>
```