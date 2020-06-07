# UCM Analysis Dash App
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Analyze and visualize data produced by the [NeuroPsy Research App](https://github.com/OlafHaag/NeuroPsyResearchApp).


## Local Setup

### Pyton Environment
* Use pip to install Pipenv
* Use Pipenv to create virtual environment
    * Use provided Pipfile
* For development, you need to install less compiler `npm install -g less`.
 Or use the slower [lesscpy](https://github.com/lesscpy/lesscpy) with [flask_less](https://github.com/mrf345/flask_less/)
 
### PostgreSQL
This app uses a PostgreSQL Database on Heroku.  
Heroku's free PostgreSQL Add-on has a row limit of 10,000.
* Install PostgreSQL (can skip Stack Builder)
* Open pgAdmin4
    * Create new user with password
        * Give privileges to login and create databases
    * Create new database with newly created user as owner
    
### Configuration
* Make a local copy of _.env-example_ and name it _.env_
* Change its values, e.g. set **DATABASE_URL** to use the values you set in the step PostgreSQL
* DO NOT SUBMIT _.env_ TO YOUR VERSION CONTROL

### Create Tables
After having set up the virtual environment with Pipenv, the database can be managed from the terminal.
* Open a terminal and navigate to the project folder.
* If using _conda_, make sure an environment with Pipenv installed is active, e.g. `conda activate base`
* `Pipenv shell` to spawn a shell with the virtualenv activated
* `flask db init` for new database.
* `flask db migrate` after changes to Model or after init. Creates versions.  
* `flask db upgrade` to apply changes and create or modify the tables in the database. **This is the only one needed to be executed on the remote db.**
* For clearing the data during development, go to a SQL prompt, or use pgAdmin or your IDE and truncate the tables cascaded. 
  Use `TRUNCATE circle_tasks, ct_sessions, devices, users CASCADE;`


## Remote Setup
### Heroku
* ...
* Set Config Vars like in _.env_
* ...
* connect repository 

### Postgres Add-on
* You can either use heroku's web-interface, or herokucli.
* `flask db upgrade` to create the tables in the database

