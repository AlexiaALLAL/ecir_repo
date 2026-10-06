#!/bin/bash
# generer_parametres.sh

# Générer les tableaux de paramètres (attention espaces non autorisés après les = )
# Réalise toutes les combinaisons des entrées du tableau parametres dont chaque valeur est une chaine dont les valeurs à tester sont séparés par des ;
# Si on veut coupler des parametres a b et c par exemple sans faire les combinaisons, on déclare parametres[a:b:c]="1:2:3;4:5:6" par exemple qui produira des configs (1,2,3) et (4,5,6) sur ces parametres sans mixer par exemple a=1 avec b=5

declare -A parametres

parametres[seed]="42"
parametres[dataset]="nq"
# if [[ "${parametres[dataset]}" == "msmarco" ]]; then
#     parametres[encoding]="prq-module"
# else
#     parametres[encoding]="pq;rq-kmeans;rq-module;pq-rq;prq-module"
# fi
parametres[encoding]="pq;rq-kmeans;rq-module;pq-rq;prq-module"
# parametres[encoding]="pq-rq" # Options: pq, rq-kmeans, rq-module, pq-rq, prq-module

# Build a coupled grid so each encoding gets its own valid (sub_space, subspace_rq, cluster_num) values.
build_encoding_grid() {
    local encoding_raw="$1"
    local encoding_values=()
    local combined=""

    IFS=';' read -r -a encoding_values <<< "$encoding_raw"

    for encoding in "${encoding_values[@]}"; do
        local group=""

        case "$encoding" in
            pq)
                group="pq:24:1:256;pq:16:1:256;pq:12:1:256;pq:8:1:256;pq:6:1:256;pq:4:1:256;pq:2:1:256"
                ;;
            rq-kmeans|rq-module)
                group="$encoding:1:24:256;$encoding:1:16:256;$encoding:1:12:256;$encoding:1:8:256;$encoding:1:6:256;$encoding:1:4:256;$encoding:1:2:256"
                ;;
            pq-rq|prq-module)
                group="$encoding:2:16:256;$encoding:2:12:256;$encoding:2:8:256;$encoding:2:6:256;$encoding:2:4:256;$encoding:2:2:256;$encoding:4:12:256;$encoding:4:8:256;$encoding:4:6:256;$encoding:4:4:256;$encoding:4:2:256;$encoding:6:4:256;$encoding:6:2:256;$encoding:8:2:256;$encoding:12:2:256"
                ;;
            *)
                echo "Erreur : encoding non supporte: $encoding" >&2
                return 1
                ;;
        esac

        if [[ -n "$combined" ]]; then
            combined+=";"
        fi
        combined+="$group"
    done

    echo "$combined"
}

encoding_grid=$(build_encoding_grid "${parametres[encoding]}") || return 1
unset 'parametres[encoding]'
parametres[encoding:sub_space:subspace_rq:cluster_num]="$encoding_grid"

getArgs() {
  local indice_job=$1
  local product=1

  # Récupérer la liste des noms des hyperparamètres
  local hyperparametres=("${!parametres[@]}")

  # --- Étape 1 : calcul des indices dans chaque tableau d’hyperparamètres
  local indices=()
  for parametre in "${hyperparametres[@]}"; do
      local raw="${parametres[$parametre]}"
      local values=()
      # découpe sur ';' pour permettre des valeurs vides
      IFS=';' read -r -a values <<< "$raw"

      local size=${#values[@]}
      local index=$(( indice_job / product % size ))
      indices+=($index)
      product=$((product * size))
  done

  echo "Liste des indices calculés pour $1 : ${indices[*]}" >&2
  

  # --- Étape 2 : récupérer les valeurs correspondantes
  local args=""
  for i in "${!hyperparametres[@]}"; do
      local parametre=${hyperparametres[$i]}
      local raw="${parametres[$parametre]}"
      local values=()
      IFS=';' read -r -a values <<< "$raw"

      local value="${values[${indices[$i]}]}"
      args+="$parametre=$value "
  done

  # supprimer l’espace final et renvoyer la ligne complète
  args="${args% }"
  echo "Liste des args calculés pour $1 : $args" >&2
  echo "$args"
}