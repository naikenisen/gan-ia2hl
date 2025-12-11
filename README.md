# IA2HL : Analyse des biopsies par Intelligence Artificielle au service des patients atteints de Lymphome de Hodgkin Avancés
    • Pr Cédric Rossi, Porteur Projet et coordination scientifique, CHU Dijon, LYSA
    • Anne-Laure BORREL, Chef de projets, coordination opérationnelle, LYSARC
    • Pr Laurent Martin, Expert anapath, CHU Dijon, LYSA 
    • Laetitia Fuhrmann, responsable LYSA-Pathologie, LYSARC 
    • Isen Naiken, Segmentation et Classification, Lymphome et IA, ImVIA / INSERM1231
    • Stéphanie Bricq, Responsable scientifique ImVIA, ImVIA LAB
    • Collègues ImViA, Segmentation (U-net, nnU-net), Intégration de données multimodales, co-encadrement thèse future,  ImVIA LAB
    • Julien Baumeyer, Classification (efficientNet, resnet), Optimisation des hyperparamètres, ImVIA LAB
    • Valentin Bosch, M2R M2SIA, Lymphome et IA, stage segmentation, UFR santé / ImVIA LAB

## Resume du projet 
Malgré la curabilité pour 80% des patients atteints de lymphome de Hodgkin (LH) classique, il est actuellement impossible de prédire de façon fiable et au diagnostic les profils de réponse et de toxicité aux stratégies de traitement actuelles. Ce projet IA2HL a pour but de répondre à ce besoin médical primordial. En effet ce projet novateur vise à identifier, dès le diagnostic, via l'analyse par intelligence artificielle (IA) des lames de biopsies, les patients à risque de rechute afin qu'ils puissent bénéficier le plus rapidement possible d'une personnalisation de leur prise en charge. C'est un projet multidisciplinaire (hématologie, anatomopathologie et techniques d'IA) impliquant plusieurs équipes et qui s'appuie sur une large base de données clinicobiologiques de 850 patients atteints d'un LH (essai AHL2011, dont le CHU de Dijon est promoteur).

## Objectifs du projet
Objectifs cliniques : Utiliser une méthode de stratification qui pourrait aider à  :

1. Objectif I (CLAM): identifier des patients réfractaires à une polychimiothérapie standard (BEACOPP, ABVD) pour une orientation le plus rapidement possible vers des stratégies intégrant de nouvelles molécules.

2. Objectif II (MODELE MULTIMODAL INTEGRATIF): identifier des associations/corrélations entre les profils de patients mis en évidence par l'IA, et les données d'imagerie TEP et biologiques (ADN tumoral circulant) déjà générées et disponibles pour les mêmes patients.

3. Objectif III : (GAN) construire des IHC synthétiques à partir des lames H&E.

## Lien vers le dépôt GitHub de CLAM:
https://github.com/mahmoodlab/CLAM

## Article de référence pour CLAM:
Lu, M. Y., Williamson, D. F. K., Chen, T. Y., Chen, R. J., Barbieri, M., & Mahmood, F. (2021). Data-efficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering, 5(6), 555–570. https://doi.org/10.1038/s41551-020-00682-w

## Lien vers le dépot kaggle du GAN:
https://www.kaggle.com/code/nibirs/ihc4bc

## Article de référence pour GAN:
Klöckner, P., Teixeira, J., Montezuma, D., Fraga, J., Horlings, H. M., Cardoso, J. S., & Oliveira, S. P. (2025). H&E to IHC virtual staining methods in breast cancer: an overview and benchmarking. NPJ digital medicine, 8(1), 384. https://doi.org/10.1038/s41746-025-01741-9

## Instructions pour le CCUB
### Alias 'pull'
```bash
alias pull='eval "$(ssh-agent -s)" && ssh-add ~/.ssh/cle && git pull'
```
### Alias 'gpu'
```bash
alias gpu='qlogin -q gpu'
```
### Alias 'st'
```bash
git status
```
### Création de l'environnement virtuel
```bash
module load python
python3 -m venv venv
source venv/bin/activate
pip3 install --prefix=/work/imvia/in156281/ia2hl/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/ia2hl/venv/lib/python3.9/site-packages:$PYTHONPATH
pip3 list
```
### Alias 'venv'
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

### Lancer le script d'entraînement
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
## lancer plusieurs jobs en parallèle avec différents hyperparamètres
```bash
bash submit_jobs.sh
```
Sachant que le script `submit_jobs.sh` fait appel à config.sh pour lancer le script `train.py` 
et que les hyperparamètres sont définis dans `config.sh`.