cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Définir le compilateur
set(CMAKE_CXX_COMPILER g++)

# Activer les warnings et options d'optimisation
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -g -std=c++11 -m64 -march=native -fopt-info-vec-optimized")

# Ajouter les fichiers source
set(CPP_FILES main.cpp) # Ajouter ici d'autres fichiers source si nécessaire
set(HXX_FILES plan.hh) # Ajouter ici les autres headers si nécessaire

# Créer une liste des fichiers objets
set(OBJ_FILES ${CPP_FILES})

# Inclure les fichiers header dans le projet (optionnel)
include_directories(${CMAKE_SOURCE_DIR})

# Spécifier les bibliothèques à lier
set(LDXX_FLAGS "-lGL -lGLEW -lglut -lpthread")

# Ajouter l'exécutable
add_executable(${PROJECT_NAME} ${CPP_FILES})

# Lier les bibliothèques
target_link_libraries(${PROJECT_NAME} ${LDXX_FLAGS})

# Skel cible pour créer une archive tar.bz2
add_custom_target(skel
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/pogl_skel_tp
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/pogl_skel_tp
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CPP_FILES} ${HXX_FILES} ${CMAKE_SOURCE_DIR}/pogl_skel_tp
        COMMAND ${CMAKE_COMMAND} -E tar cfjv pogl_skel_tp.tar.bz2 pogl_skel_tp
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/pogl_skel_tp
)

# Messages de compilation
add_custom_target(pre-build
        COMMAND ${CMAKE_COMMAND} -E echo "******** Starting Compilation ************"
)

add_custom_target(post-build
        COMMAND ${CMAKE_COMMAND} -E echo "*********** End Compilation **************"
)
