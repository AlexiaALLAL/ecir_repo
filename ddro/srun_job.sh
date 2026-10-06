#!/bin/bash

pwd

# Récupérer l'indice de job passé en paramètre
indice_job=$1

#args=$(getArgs $indice_job)

find_config(){
  id=$1
  fic_csv=$2
  IFS=, read -ra headers < "$fic_csv"
  while IFS=, read -r -a line; do
      # Vérifier si la valeur de la colonne A correspond à l'ID recherché
      if [ "${line[0]}" == "$id" ]; then
          # Construire la chaîne B=b_id C=c_id D=d_id
          colonnes=""

          for ((i=0; i<${#line[@]}; i++)); do
              # Exclure la colonne A
              if [ "$i" != "0" ]; then
                  colonnes+=" ${headers[i]}=${line[i]}"
              fi
          done

          # Afficher la chaîne des colonnes
          echo "${colonnes:1}"
          return 0
      fi
  done < "$fic_csv"
  # Si aucune ligne correspondante n'est trouvée
  echo "Aucune ligne trouvée avec id config=$id"
  return 1
}

config=$(find_config $indice_job "./configs.csv")
echo "config de $indice_job = $config"

# Afficher les valeurs d'hyperparamètres
echo "Hyperparamètres du job $indice_job : ${args[@]}"

tmp_dir=""
if [[ "$5" != "" ]]; then
  varname="$5"
  value="${!varname}"
  echo "$value"
  tmp_dir="tmp_dir=$value"
fi
    

# Lancer votre_script.py avec les valeurs d'hyperparamètres correspondantes
#python votre_script.py "${args[@]}"
#python -u train.py "job='$2_$1'" "${args[@]}" "output=$3"
cmd="job='$2_$1' $config output='$3' $tmp_dir ref_dir='$4'"
# echo "commande = python -u main.py $cmd"
#python -u train.py "job='$2_$1' " " $config " "output=$3"

# python -u main.py ${cmd}


# Convert "key=value" format to "--key value" format
script_args=""
for pair in $config; do
    key=$(echo "$pair" | cut -d'=' -f1)
    value=$(echo "$pair" | cut -d'=' -f2)
    script_args+="--$key $value "
done

script_name="src/scripts/jean-zay/train_full_pipeline.sh"

echo "Running command: bash $script_name $script_args"
bash $script_name $script_args