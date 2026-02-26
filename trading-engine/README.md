# Trading Engine

This repository hosts the **Production Engineering** & **Research Engineering** parts of CapitalFund. Follow the steps below to set up your local development environment and start contributing.

---

## 📋 Prerequisites

Before you begin, ensure you have the following:

* **Docker** installed ([Download here](https://docs.docker.com/get-docker/)).
* An **IDE** of your choice (e.g., [PyCharm](https://www.jetbrains.com/pycharm/) or [VS Code](https://code.visualstudio.com/)).
* A **GitHub Account** with access to the organization.
* A valid **`service-account.json`** file (provided by an admin).
* Access granted to the **GitHub Organization**.

---

## 🚀 Setup Instructions

1. **Request Access**
   Contact a repository admin to request access to the GitHub organization and this repository.

2. **Clone the Repository**
   Once you have access, clone the repo:

   ```bash
   git clone https://github.com/<org-name>/trading-engine.git
   cd trading-engine
   ```

   > If you are having trouble cloning the repository, trying using the [GitHub CLI](https://cli.github.com/)

3. **Configure Service Account**
   Place your `service-account.json` in a secure location. Export the path as an environment variable:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   ```

## 📚 Additional Resources

* [Docker Documentation](https://docs.docker.com/)
* [GitHub Docs](https://docs.github.com/)
* [Google Cloud Authentication](https://cloud.google.com/docs/authentication/getting-started)


## Poetry Instructions

> [!NOTE]
> The following instructions are for macOS/Linux machines and may not work exactly as expected on Windows systems.


to install the dependencies using poetry, run the following
```bash
poetry install
```

To register the poetry enviroment as a jupyter kernel (for use with IDE notebook editors) 
(this installs the kernel in a user specific location)

```bash
poetry run python -m ipykernel install --user --name trading-engine --display-name "Python (poetry) trading-engine"
```
(Optional) Check that the jupyter kernel installed properly
```bash
poetry run jupyter kernelspec list
```

After registering the kernel, you may need to restart your IDE (vscode, cursor) for it to display as an option.

> [!NOTE]
> When running notebook code via your IDE configuration, you may encounter an error related to google cloud storate.

To fix the GCS issue, you'll need to export the enviroment varible within the notebook.
Run a notebook cell with this line of code.

```python
# Set Google Application Credentials Enviroment (Only needed for 'local' poetry mode, not docker)

# Use Absolute Path for Google Application Credentials
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'abs_path_to_your_service_account/service_account.json' # CHANGE THIS PATH
```


## Mosek Opt Liscense
```bash
export MOSEKLM_LICENSE_FILE=/path_to_lic_file/mosek.lic
```
