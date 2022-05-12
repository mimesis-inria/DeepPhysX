## DeepPhysX

### Interfacing AI with simulation

The **DeepPhysX** project provides Python packages allowing users to easily interface their **numerical simulations**
with **learning algorithms**.

**DeepPhysX** provides a **Core** package with no dependency on any **simulation** or **AI** framework.
Then other modules are compatible with this **Core** and a specific **simulation** or **AI** framework.


### Quick install

The project was initially developed using [SOFA](https://www.sofa-framework.org/) as the **simulation package** and
[PyTorch](https://pytorch.org/) as the **AI framework**. 
Thus, **DeepPhysX** is mainly designed for these frameworks, but obviously **other frameworks** can also be used.
The packages corresponding to these frameworks will therefore be used for the default installation.
For further instructions (set up your installation config, developer mode), please refer to the **documentation**.

``` bash
$ git clone https://github.com/mimesis/deepphysx.git
$ cd DeepPhysX
$ pip install .
```
