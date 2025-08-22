# Numerical Geometry

A numerical geometry project in Python, with a focus on mesh deformations using neural-networks.

## Installation

This project is designed to be explored through Jupyter notebooks. To set up:

```bash
# Clone and navigate to the project.
git clone <https://github.com/joeEndacott/numerical_geometry.git>
cd numerical_geometry

# Install in development mode.
pip install -e .

# Explore the notebooks.
jupyter lab notebooks/
```

## Notebooks

- **`cube.ipynb`** - generating a cube from scratch using PyVista.
- **`hemisphere_to_cow.ipynb`** - training a neural network to deform a hemisphere into a cow's head.
- **`sphere_to_cube.ipynb`** - training a neural network to deform a sphere into a cube.
- **`sphere.ipynb`** - generating a sphere from scratch using PyVista.

## Project organization

```
numerical_geometry/
├── numerical_geometry/     # Core utilities
│   ├── core.py
│   └── utils.py
└── notebooks/              # Exploration and experiments
    ├── cube.ipynb
    ├── hemisphere_to_cow.ipynb
    ├── questions.ipynb
    ├── sphere_to_cube.ipynb
    └──  sphere.ipynb
```

## Disclaimer

This code is experimental and educational in nature. It may contain bugs, inefficiencies, or unconventional approaches as it represents a learning journey rather than a production codebase.
