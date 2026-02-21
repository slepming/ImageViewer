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
  echo "w-r/w-d (windows release, windows debug)"
  read -p "n-r/n-d/c?(default run) " version
  if [ "$version" = "n-r" ]; then
    cargo build --release
  elif [ "$version" = "c" ]; then
    cargo clean
  elif [ "$version" = "n-d" ]; then
    cargo build
  elif [ "$version" = "w-r" ]; then
    echo "windows release building for x86_64-pc-windows-gnu"
    cargo build --release --target x86_64-pc-windows-gnu
  elif [ "$version" = "w-d" ]; then
    echo "windows building for x86_64-pc-windows-gnu"
    cargo build --target x86_64-pc-windows-gnu
  else
    cargo run "$path_to_image"
  fi
}

everyone() {
  compile_shaders
  project_building
}

everyone
