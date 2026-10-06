#!/bin/bash

# pour lancer en local ajouter --local a la ligne de commande
# pour jeanzay en mode dev pour tests courts ajouter --dev a la ligne de commande

nom_projet="ddro"  # sur jeanzay ==> suppose que l'on a créé un venv du nom du projet via : python -m venv $WORK/envs/$nom_projet

nom_xps=$1
shift

# Valeurs par défaut
local_mode=false
debug_mode=false
continue_xps=""
tmp_dir=""
timeout=""
# Boucle de parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            local_mode=true
            shift
            ;;
        --dev)
            debug_mode=true
            shift
            ;;
        --continue_xps)  # exemple : sh launcher.sh continue_test_protect_critic --continue_xps test_protect_critic
            # Vérifie qu'un argument suit
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                continue_xps="$2"
                shift 2  # on consomme les deux arguments
            else
                echo "Erreur : l’option --continue_xps nécessite un argument (ex: --continue_xps nom_xp)"
                exit 1
            fi
            ;;
        --timeout)
            # Vérifie qu'un argument suit
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                timeout="$2"
                shift 2  # on consomme les deux arguments
            else
                echo "Erreur : l’option --timeout nécessite un argument (ex: --timeout 00:10:00)"
                exit 1
            fi
            ;;

        *)
            echo "Option inconnue : $1"
            shift
            ;;
    esac
done


# Affichage pour vérif
echo "Projet : $nom_projet"
echo "Nom Xps : $nom_xps"
echo "local mode : $local_mode"
echo "Debug : $debug_mode"
echo "Continue XPs : $continue_xps"


specif_nodes=""
specif_ntask_per_node=""
specif_gres=""
specif_cpus_per_task=""

ref_dir=$(pwd)
if $local_mode; then
    racine=".."
    nb_simultaneous=1
else
    racine=$WORK/$nom_projet #$SCRATCH/$nom_projet
    if $debug_mode; then   # pour debug rapide
        tmp_dir="JOBSCRATCH"
        nb_simultaneous=10   # nombre max de jobs parallèles
        nb_nodes=1
        nb_gpu=1
        nb_cpu=4
        # if timeout is ""
        if [[ -z "$timeout" ]]; then
            timeout="02:00:00"
            # timeout="00:10:00"
        fi
        specif_gpu=""
        specif_partition=""    # partition libre
        qos="qos_gpu_a100-dev"
    else
        tmp_dir="JOBSCRATCH"
        nb_simultaneous=90   # nombre max de jobs parallèles
        nb_nodes=1
        nb_gpu=1
        nb_cpu=8
        if [[ -z "$timeout" ]]; then
            timeout="20:00:00"
        fi
        qos="qos_gpu_a100-t3"
    fi
    specif_nodes="SBATCH --nodes=$nb_nodes"
    specif_ntask_per_node="SBATCH --ntasks-per-node=$nb_gpu"
    specif_gres="SBATCH --gres=gpu:$nb_gpu"
    specif_cpus_per_task="SBATCH --cpus-per-task=$nb_cpu"
    specif_gpu="SBATCH -C a100"
fi
racine=$(realpath $racine)
echo "Racine : $racine"

pwd
source ./param_grid.sh  # chargement des listes d'hyperparametres a tester



# Fonction pour créer un répertoire avec ses répertoires parents si nécessaire
mkdirs() {
    local directory="$1"
    # Vérifier si le répertoire existe déjà
    if [ ! -d "$directory" ]; then
        # Créer le répertoire et ses parents avec l'option -p
        mkdir -p "$directory"
        echo "Répertoire créé : $directory"
        #chmod 755 "$directory"
        #chmod +x mon_script.sh

        return 0 # Indique que le répertoire a été créé
    else
        echo "Le répertoire $directory existe déjà. Arrêt du programme."
        exit 1 # Arrête le programme avec un code d'erreur
    fi
}



xp_dir="$racine/xp/$nom_xps/"
echo "XPs dir : $xp_dir"

#xp_dir="./test"



mkdirs $xp_dir
#echo "repertoire $xp_dir créé"

#cp -Rf ./* $xp_dir

rsync -av --progress --exclude='exclude_from_copy' --exclude='.git' --exclude='__pycache__' ./ "$xp_dir"

cd $xp_dir
ici=$(pwd)
echo "On se trouve dans $ici"

chmod u+w .
fichier_csv="./configs.csv"
#echo "eee" >> "$fichier_csv"

hyperparametres=("${!parametres[@]}")

# Nombre de combinaisons de parametres
product=1
for parametre in "${hyperparametres[@]}"; do
      raw="${parametres[$parametre]}"
      values=()
      # découpe sur ';' pour permettre des valeurs vides
      IFS=';' read -r -a values <<< "$raw"

      size=${#values[@]}
      if (( size > 1 )); then
        if [[ $continue_xps ]]; then
            echo "Erreur : continue xps est incompatible avec des listes de valeurs dans param_grid : une seule valeur autorisée par paramètre : on continue les xps mentionnées avec leur config exacte, sauf en remplaçant les paramètres indiqués par la valeur donnée. " >&2
            exit 1
        fi
      fi
      
      product=$((product * size))
done

echo "$product jobs a lancer"


#$(create_fic_configs $product $fichier_csv)

#chmod 755 $fichier_csv
header=""
last_vals=""
for ((i = 0; i < product; i++));
do

    args=()
    #echo "$i $args"
    IFS=' ' read -r -a args <<< "$(getArgs $i)"
    cles=""
    valeurs=""

    for paire in "${args[@]}"; do
        # Séparation de la paire clé-valeur en utilisant le signe "=" comme délimiteur
        cle_brute=$(echo "$paire" | cut -d '=' -f 1)
        valeur_brute=$(echo "$paire" | cut -d '=' -f 2)

        # découpe sur :
        IFS=':' read -r -a subcles <<< "$cle_brute"
        IFS=':' read -r -a subvaleurs <<< "$valeur_brute"

        # on parcourt chaque sous-clé avec sa sous-valeur
        for ((j=0; j<${#subcles[@]}; j++)); do
            cle="${subcles[$j]}"
            valeur="${subvaleurs[$j]}"
            cles+=",$cle"
            valeurs+=",$valeur"
        done
    done

    if [ "$i" -eq 0 ]; then
        header="config$cles"
        echo "config$cles" >> "$fichier_csv"
    fi
    echo "$i$valeurs" >> "$fichier_csv"
    last_vals="$i$valeurs"
done

# pour relancer des xps arretées trop tôt (en modifiant quelques paramètres via param_grid éventuellement)
if [[ $continue_xps ]]; then
    tmp_csv="./tmp_configs.csv"
    product=0
    # Création du fichier de sortie
    echo "${header},load_params_from,protagonist_dir,q_dir" > "$tmp_csv"

    echo "Search job directories in $racine/xp/$continue_xps/results/"
    # Boucle sur les sous-répertoires
    shopt -s nullglob
    for dir in $racine/xp/$continue_xps/results/*; do
        echo "Found $dir"
        if [ ! -d "$dir" ]; then
            echo "Not a directory, continue"
            continue
        fi

        dirname=$(basename "$dir")
        echo "dirname = $dirname"
        # Vérifie que le nom contient un "_"
        if [[ "$dirname" != *_* ]]; then
            echo "Ignoré : $dirname"
            continue
        fi

        prefix="${dirname%%_*}"  # avant le _
        config="${dirname##*_}"  # après le _

        # Remplacement du champ 'config' (premier champ) en conservant le reste
        rest="${last_vals#*,}"
        new_line="${config},${rest}"
        # récupérer le chemin complet
        full_path=$(realpath "$dir")  # ou $(pwd)/"$dir" si realpath n'est pas disponible


        # Ajout de la colonne "from"
        echo "${new_line},${full_path}/setter_conf.yaml,${full_path}/models/protagonist,${full_path}/models/q_nets" >> "$tmp_csv"
        product=$((product + 1))
    done
    mv "$tmp_csv" "$fichier_csv"
    
fi

echo "$product jobs a lancer"


#exit 1

last_idx=$((product - 1))
#sbatch jz_submit_multi.slurm $product



# -----------------------------
# Mode local ou Slurm
# -----------------------------
if $local_mode; then
    echo "🔧 Running in LOCAL mode..."

    # Limite de jobs en parallèle (par ex. 8 si ta machine a 8 cœurs)
    max_parallel=$nb_simultaneous
    running_jobs=0

    pids=()

    cleanup() {
        echo "💥 Ctrl+C détecté ! Killing jobs..."
        for pid in "${pids[@]}"; do
            # tuer tous les enfants du PID
            pkill -TERM -P "$pid" 2>/dev/null
            # tuer le PID lui-même
            kill -TERM "$pid" 2>/dev/null
        done
        exit 1
    }

    trap cleanup SIGINT

    for ((i=0; i<product; i++)); do
        ./srun_job.sh "$i" "local" "$nom_xps" "$ref_dir" "$tmp_dir" &  # lancement en tâche de fond
        pid=$!
        pids+=($pid)
        ((running_jobs++))

        # Si on atteint la limite, attendre qu’un job se termine
        if (( running_jobs >= max_parallel )); then
            wait -n
            ((running_jobs--))
        fi
    done

    # Attendre la fin de tous les jobs restants
    wait
    echo "✅ Tous les jobs locaux sont terminés."

else

chmod +x srun_job.sh


# Build optional SBATCH directives
sbatch_qos=""

# Jean Zay presets
[ -n "$qos" ] && sbatch_qos="#SBATCH --qos=$qos"

tmpfile=$(mktemp)

cat <<EOT > "$tmpfile"
#!/bin/bash
##SBATCH --export=ALL,MASTER_PORT=23456
#SBATCH --job-name=$nom_xps         # nom du job
#$specif_gpu
#$specif_partition
#$specif_nodes                       # on demande un noeud
#$specif_ntask_per_node              # avec une tache par noeud (= nombre de GPU ici)
#$specif_gres                        # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#$specif_cpus_per_task               # nombre de CPU par tache
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=$timeout              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=gpu_$nom_projet%x_%A_%a.out      # nom du fichier de sortie
#SBATCH --error=gpu_$nom_projet%x_%A_%a.out       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-$last_idx%$nb_simultaneous
$sbatch_qos
#SBATCH --account=coa@a100



# Nettoyage des modules charges en interactif et herites par defaut
module purge


# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition
# module load cpuarch/amd

# Chargement des modules
#module
#load...
#conda deactivate
#module load pytorch-gpu/py3/2.1.1
#conda init bash


module load python/3.10.4
module load arch/a100
source $WORK/envs/$nom_projet/bin/activate

# Echo des commandes lancees
set -x

# Récupérer le numéro de job
job_id=\$SLURM_JOB_ID

# Pour la partition "gpu_p5", le code doit etre compile avec les modules compatibles
# Execution du code
srun ./srun_job.sh \${SLURM_ARRAY_TASK_ID} \${SLURM_ARRAY_JOB_ID} $nom_xps $ref_dir $tmp_dir


EOT

echo "===== SCRIPT SBATCH GÉNÉRÉ ====="
cat "$tmpfile"
echo "================================"

# Submit main job array and capture job ID
MAIN_JOB_ID=$(sbatch --parsable "$tmpfile")
echo "Submitted main job array: $MAIN_JOB_ID"

# For full_training jobs, submit a restart checker that runs after the array completes
# Check if this is a full_training run by looking at the configs
FIRST_CONFIG=$(sed -n '2p' "$fichier_csv")
echo "This is a full_training job - setting up automatic restart checker"

# Create a restart checker script that processes all array tasks
restart_tmpfile=$(mktemp)
cat <<RESTART_EOT > "$restart_tmpfile"
#!/bin/bash
#SBATCH --job-name=${nom_xps}_restart_check
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=restart_check_%x_%A.out
#SBATCH --error=restart_check_%x_%A.out
#SBATCH --qos=$qos
#SBATCH --account=coa@a100

module purge
module load python/3.10.4
module load arch/a100
source \$WORK/envs/$nom_projet/bin/activate

cd $xp_dir

echo "=== Checking which jobs need to be restarted ==="

# Process each config to check if restart is needed
for i in \$(seq 0 $last_idx); do
    bash src/scripts/jean-zay/restart_if_needed.sh "$MAIN_JOB_ID" "\$i" "$nom_xps" "$ref_dir" "$tmp_dir" "$nom_projet" "$qos" "$timeout"
done

echo "=== Restart check completed ==="
RESTART_EOT

    RESTART_JOB_ID=$(sbatch --dependency=afterany:$MAIN_JOB_ID --parsable "$restart_tmpfile")
    echo "Submitted restart checker job: $RESTART_JOB_ID (will run after main job completes)"
    rm "$restart_tmpfile"

fi