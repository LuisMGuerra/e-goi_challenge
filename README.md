# E-goi Technical Challenge

This small report details both the structure of the scripts I produced to solve the challenge and the decisions I made during coding.

## Overview

The main script were the model was researched and developed is called *model_research.py* and should be run first as it produces the model and data sample necessary to test the deployment REST API.
Once that script is run and the files *final_model.sav* and *request_sample.csv* are created the script *flask_server_side.py* can be run, simulating the server, and the script *flask_client_side.py* can be run to simulate a call to the API where it asks for a prediction for the sample in *request_sample.csv* done by the model *final_model.sav* on the server side.

### Prerequisites

The script was developed on an anaconda environment but should run on any python 3.7 installation provided, numpy, pandas, scikit-learn, matplotlib, seaborn and flask are installed.
Numpy and Pandas were used for general data wrangling, maplotlib and seaborn for visualizations, scikit for modelling and flask to create the deployment REST API.

## Preprocessing

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
