# Project de ISIM : Génération de texture procédurale

## Build et execution sur terminal:

### Build:
```shell
mkdir build && cd build
cmake ..
make
```

### Run:
```shell
#En suposant que vous etes encore dans le fichier build
./ISIM72 {filename}.JSON
```

## Formation du JSON base:

La structure de base du JSON est une liste, vous pouvez donc generer autant de texture en une execution que vous voulez.
```json
[
  {
    ...
  },
  {
    ...
  }
]
```
Pour initialiser une nouvelle, il faut quatre mots cles principaux:
```json
{
  "saveAs": string
  "size": {
    "width": int
    "height": int
  },
  "type": string
  "parametre": {
    ...
  }
}
```
  * "size": un object, definit la taille final de la texture.
    * "width": int, largeur de la texture
    * "height": int, hauteur de la texture

  * "type": string, precise quelle texture est attendu.

  * "parametre": object, regroupe tous les parametres a choisir pour generer une texture souhaiter.

  * "saveAs": string, n'est pas obligatoire, mais permet de sauvegarder la texture generer au format image demander. (Il est recommender d'inscrire le PATH absolue pour etre sur de son emplacement)

## Formation du JSON parametre:

<b><u>Note:
- Toutes les couleurs sont encode en hexadecimale sous le format :  "#XXXXXX" ou "#XXXXXXXX" pour changer absorption
- Une couleur null et une couleur Transparente (r=0, g=0, b=0, a=0)
- Pour certain parametre de couleur telle que dans "space", certainne fentesies sont possible (voir les exemples pour plus de detail)
</u></b>
### Gradiant:

```json
"type": "gradient"
"parametre": {
  "color": [string]
  "angle": int
}
```
* "color": liste de string, couleur a insterpoller entre elle pour former un gradiant.
* "angle": int, sur quelle angle, en degre, l'interpolation aura lieu.

### Brick:

```json
"type": "brick"
"parametre": {
  "brick": {
    "size": {
      "width": int
      "height": int
    },
    "color": string
  },

  "space": {
    "len": int
    "color": string
  }
}
```
* "brick": object, definit les details de la brick
  * "size": object,  definit les dimention d'une brick
    * "width": int, largeur de la brick
    * "height": int, hauteur de la brick
  * "color": string, definit la couleur des bricks
* "space": object, definit l'espacement entre deux bricks
  * "len": int, taille de l'espacement
  * "color": string, couleur entre les bricks

### Pavet de Claire

```json
"type": "claire",
"parametre": {
  "a": int,
  "b": int,
  "up": string,
  "down": string,
  "left": string,
  "right": string
  "space": {
    "len": 5,
    "color": string
      }
}
```
* "a": int, taille du pentagon (voir fig)
* "b": int, taille du pentagon (voir fig)
* "up": string, couleur du pentagonne haut
* "down": string, couleur du pentagonne bas
* "left": string, couleur du pentagonne gauche
* "right": string, couleur du pentagonne droit
* "space": object, definit l'espacement entre deux pavet
    * "len": int, taille de l'espacement
    * "color": string, couleur entre les pavet


<img alt="aide pour setup a et b pour le pavet de Claire" height="250" src="/readme_image/claire_shape.png" title="clair_dim" width="250"/>

## JSON exemple:

### Gradient, Arc en ciel

```json
[
  {
    "saveAs": "gradient.png",
    "size": {
      "width": 1920,
      "height": 1080
    },
    "type": "gradient",
    "parametre": {
      "color": [
        "#FF0000",
        "#FF7F00",
        "#FFFF00",
        "#00FF00",
        "#0000FF",
        "#4B0082",
        "#9400D3"
      ],
      "angle": 45
    }
  }
]
```
### Brick classic

```json
[
  {
    "saveAs": "brick_classic.png",
    "size": {
      "width": 1920,
      "height": 1080
    },
    "type": "brick",
    "parametre": {
      "brick": {
        "size": {
          "width": 50,
          "height": 20
        },
        "color": "#FF0000"
      },
      "space": {
        "len": 5,
        "color": "#000000"
      }
    }
  }
]
```

### Brick avec gradient

```json
[
  {
    "saveAs": "brick_gradient.png",
    "size": {
      "width": 1920,
      "height": 1080
    },
    "type": "brick",
    "parametre": {
      "brick": {
        "size": {
          "width": 50,
          "height": 20
        },
        "parametre": {
          "saveAs": "brick_gradient_etape.png",
          "size": {
            "width": 1920,
            "height": 1080
          },
          "type": "gradient",
          "parametre": {
            "color": [
              "#FF0000",
              "#FF7F00",
              "#FFFF00",
              "#00FF00",
              "#0000FF",
              "#4B0082",
              "#9400D3"
            ],
            "angle": 45
          }
        }
      },
      "space": {
        "len": 5,
        "color": "#000000"
      }
    }
  }
]
```

### Pave de Claire

```json
[
  {
    "saveAs": "Claire.png",
    "size": {
      "width": 1920,
      "height": 1080
    },
    "type": "claire",
    "parametre": {
      "a": 20,
      "b": 40,
      "up": "#0000FF",
      "down": "#FF0000",
      "left": "#FFFF00",
      "right": "#00FF00",
      "space": {
        "len": 5,
        "color": null
      }
    }
  }
]
```

### Pave de Claire LGBT

Note : la parametre.[up|down|left|right].size sera calculer automatiquement selon a et b.

```json
[
    {
      "saveAs": "Claire_gradient.png",
      "size": {
        "width": 1920,
        "height": 1080
      },
      "type": "claire",
      "parametre": {
        "a": 20,
        "b": 40,
        "up": {
          "saveAs": "Claire_gradient_UP.png",
          "size": {
            "width": 80,
            "height": 80
          },
          "type": "gradient",
          "parametre": {
            "color": [
              "#FF0000",
              "#FF7F00",
              "#FFFF00",
              "#00FF00",
              "#0000FF",
              "#4B0082",
              "#9400D3"
            ],
            "angle": 45
          }
        },
        "down": {
          "saveAs": "Claire_gradient_DOWN.png",
          "size": {
            "width": 80,
            "height": 80
          },
          "type": "gradient",
          "parametre": {
            "color": [
              "#FF0000",
              "#FF7F00",
              "#FFFF00",
              "#00FF00",
              "#0000FF",
              "#4B0082",
              "#9400D3"
            ],
            "angle": 90
          }
        },
        "left": {
          "saveAs": "Claire_gradient_LEFT.png",
          "size": {
            "width": 80,
            "height": 80
          },
          "type": "gradient",
          "parametre": {
            "color": [
              "#FF0000",
              "#FF7F00",
              "#FFFF00",
              "#00FF00",
              "#0000FF",
              "#4B0082",
              "#9400D3"
            ],
            "angle": 135
          }
        },
        "right": {
          "saveAs": "Claire_gradient_RIGHT.png",
          "size": {
            "width": 80,
            "height": 80
          },
          "type": "gradient",
          "parametre": {
            "color": [
              "#FF0000",
              "#FF7F00",
              "#FFFF00",
              "#00FF00",
              "#0000FF",
              "#4B0082",
              "#9400D3"
            ],
            "angle": 180
          }
        },
        "space": {
          "len": 5,
          "color": "#000000"
        }
      }
    }
]
```

### Psyco:
```json
[
  {
    "saveAs": "psyco.png",
    "size": {
      "width": 1920,
      "height": 1080
    },
    "type": "bois3",
    "parametre": {
      "mid": "#8B4513",
      "inf": "#FF00FF",
      "up": "#8B4513",
      "borne_up": 0.7,
      "borne_inf": 0.6
    }
  }
]
```

### Return code :

- 0: All good
- -1: JSON error
- -2: Color error
- -3: Size error
