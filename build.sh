#!/bin/sh

shader_path=./shaders

path_to_image=$1

compile_shaders() {
  echo "building shaders."
  if [ -d "$shader_path" ]; then
    for shader in "$shader_path"/*; do
      if [[ "${shader}" == *.frag || "${shader}" == *.vert ]]; then
        glslangValidator -V "$shader" -o "$shader.spv"
      fi
    done
  else
    question='y'
    read -p "shader path not existed.. Create directory? [y/n]: " question
    if [ "$question" = 'y' ]; then
      mkdir $shader_path
    fi
    compile_shaders
  fi
}

project_building() {
  echo "building project.."
  version="debug"
  read -p "release/debug/clean?(default run) " version
  if [ "$version" = "release" ]; then
    cargo build --release
  elif [ "$version" = "clean" ]; then
    cargo clean
  elif [ "$version" = "debug" ]; then
    cargo build
  else
    cargo run "$path_to_image"
  fi
}

everyone() {
  compile_shaders
  project_building
}

everyone
