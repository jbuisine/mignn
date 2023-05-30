# MiGNN

## Description

MiGNN is a library for processing light paths and composing graphs. The idea of this tool is to train deep learning models from simplified scene graphs.

## Installation

This project depends on Mitsuba3 renderer. First, clone the specific mitsuba version for light paths extraction:
```
git clone --resursive -b gnn-integrator https://github.com/Artisis-ULCO/mitsuba3.git
```

**Note:** before compiling project ensure you are in your prefered Python environnement (virtual or specific Python version). Mitsuba Python binder will be associated to this specific Python version.

Then compile the project:
```
cd mitsuba3
mkdir build && cd build
cmake -GNinja ..
ninja
```

When you need to load  and/or update `mitsuba3` binaries:
```
source build/setpath.sh
```

## Usage

In order to extract light paths, you need to specify the expected `pathgnn` plugin and output data file in your scene file:

```xml
<scene version="3.0.0">
	<default name="integrator" value="pathgnn" />
    <string name="logfile" value="gnn_file.path"/>
    ...
</scene>
```

### Generate light paths data
You can dynamically render from multiple viewpoints with a respective ligthpath data file:
```python
import drjit as dr
import mitsuba as mi
mi.set_variant('scalar_rgb')

# load scene
scene = mi.load_file("scenes/simple.xml")

# update output file
params = mi.traverse(scene)
params['logfile'] = 'gnn_file_1.path'
params.update()

# check update
print(params['logfile'])
```
**Note:** `scalar_rgb` is mandatory in order to well log data.

### Manage graphs and increase knowledge

Load your data:
```python
from mignn.container import SimpleLightGraphContainer
light_graphs = SimpleLightGraphContainer.fromfile('data/gnn_file_1.path', scene_file, verbose=True)
>>> SimpleLightGraphContainer: [n_keys: 4096, n_graphs: 40960, n_nodes: 138231 (duplicate: 0), n_connections: 97271 (built: 0)]
```

From the light graphs (obtained from multiple samples) increase knowledge:

```python
light_graphs.build_connections(n_graphs=10, n_nodes_per_graphs=2, n_neighbors=5, verbose=True)
>>> SimpleLightGraphContainer: [n_keys: 4096, n_graphs: 81920, n_nodes: 313891 (duplicate: 37429), n_connections: 232438 (built: 37896)]
```

## LICENSE

[MIT](LICENSE)