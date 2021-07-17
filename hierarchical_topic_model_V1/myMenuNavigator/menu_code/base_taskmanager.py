import shutil
import _pickle as pickle
import yaml

import pathlib
import logging


class baseTaskManager(object):
    """
    Base Task Manager class.

    This class provides the basic functionality to create, load and setup an
    execution project from the main application

    The behavior of this class might depend on the state of the project, which
    is stored in dictionary self.state, with the followin entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file succesfully loaded and processed
    """

    def __init__(self, path2project, path2source=None,
                 config_fname='parameters.yaml',
                 metadata_fname='metadata.pkl', set_logs=True):
        """
        Sets the main attributes to manage tasks over a specific application
        project.

        Parameters
        ----------
        path2project : str or pathlib.Path
            Path to the application project
        path2source : str or pathlib.Path or None, optional (default=None)
            Paht to the folder containing the data sources for the application.
            If none, no source data is used.
        config_fname : str, optional (default='parameters.yaml')
            Name of the configuration file
        metadata_fname : str or None, optional (default=None)
            Name of the project metadata file.
            If None, no metadata file is used.
        set_logs : bool, optional (default=True)
            If True logger objects are created according to the parameters
            specified in the configuration file
        """

        # This is the minimal information required to start with a project
        self.path2project = pathlib.Path(path2project)
        self.path2metadata = self.path2project / metadata_fname
        self.path2config = self.path2project / config_fname
        if path2source is not None:
            self.path2source = pathlib.Path(path2source)

        # Metadata attributes
        self.metadata_fname = metadata_fname

        # Dictionary of parameters that will be read from the configuration
        # file
        self.global_parameters = None

        # These are the default file and folder names for the folder
        # structure of the project. It can be modified by entering other
        # names as arguments of the create or the load method.
        self.f_struct = {}

        # State variables that will be loded from the metadata file when
        # when the project was loaded.
        self.state = {
            'isProject': False,     # True if the project exist.
            'configReady': False}   # True if config file has been processed

        # The default metadata dictionary only contains the state dictionary.
        self.metadata = {'state': self.state}

        # Other class variables
        self.ready2setup = False  # True after create() or load() are called

        # Logger object (that will be activated by _set_logs() method)
        self.set_logs = set_logs
        self.logger = None

        return

    def _set_logs(self):
        """
        Configure logging messages.
        """

        # Log to file and console

        p = self.global_parameters['logformat']
        fpath = self.path2project / p['filename']

        if not self.logger:
            logging.basicConfig(
                filename=fpath, format=p['file_format'],
                level=p['file_level'], datefmt=p['datefmt'], filemode='a')

            # Define a Handler to write messages to the sys.stderr
            console = logging.StreamHandler()
            console.setLevel(p['cons_level'])
            # Set a format which is simpler for console use
            formatter = logging.Formatter(p['cons_format'])
            # Tell the handler to use this format
            console.setFormatter(formatter)
            # Add the handler to the root logger
            self.logger = logging.getLogger()
            self.logger.addHandler(console)
            # logging.getLogger('').addHandler(console)

        return

    def _update_folders(self, f_struct=None):
        """
        Creates or updates the project folder structure using the file and
        folder names in f_struct.

        Parameters
        ----------
        f_struct: dict or None, optional (default=None)
            Contains all information related to the structure of project files
            and folders:
                - paths (relative to the project path in self.path2projetc)
                - file names
                - suffixes, prefixes or extensions that could be used to define
                  other files or folders.
            If None, names are taken from the current self.f_struct attribute
        """

        # ######################
        # Project file structure

        # Overwrite default names in self.f_struct dictionary by those
        # specified in f_struct
        if f_struct is not None:
            self.f_struct.update(f_struct)

        # In the following, we assume that all files in self.f_struct are
        # subfolders of self.path2project. If this is not the case, this method
        # should be modified by a child class
        for d in self.f_struct:
            path2d = self.path2project / self.f_struct[d]
            if not path2d.exists():
                path2d.mkdir()

        return

    def _save_metadata(self):
        """
        Save metadata into a pickle file
        """

        # Save metadata
        with open(self.path2metadata, 'wb') as f:
            pickle.dump(self.metadata, f)

        return

    def _load_metadata(self):
        """
        Load metadata from a pickle file

        Returns
        -------
            metadata : dict
                Metadata dictionary
        """

        # Save metadata
        print('-- Loading metadata file...')
        with open(self.path2metadata, 'rb') as f:
            metadata = pickle.load(f)

        return metadata

    def setup(self):
        """
        Sets up the application projetc. To do so, it loads the configuration
        file and activates the logger objects.
        """

        # #################################################
        # Activate configuration file and load data Manager
        print("\n*** ACTIVATING CONFIGURATION FILE")

        if self.ready2setup is False:
            exit("---- Error: you cannot setup a project that has not been " +
                 "created or loaded")

        with open(self.path2config, 'r', encoding='utf8') as f:
            self.global_parameters = yaml.safe_load(f)

        # Set up the logging format
        if self.set_logs:
            self._set_logs()

        self.state['configReady'] = True

        # Sace the state of the project.
        self._save_metadata()
        
        print("---- Configuration file activated.")
        
        return
    
    def create(self):
        """
        Creates an application project.
        To do so, it defines the main folder structure, and creates (or cleans)
        the project folder, specified in self.path2project
        """

        print("\n*** CREATING NEW PROJECT")

        # #####################
        # Create project folder

        # This is just to abbreviate
        p2p = self.path2project
        p2c = self.path2config

        # Check and clean project folder location
        if p2p.exists():
            print(f'Folder {p2p} already exists.')

            # Remove current backup folder, if it exists
            old_p2p = p2p.parent / (p2p.name + '_old/')
            if old_p2p.exists():
                shutil.rmtree(old_p2p)

            # Copy current project folder to the backup folder.
            shutil.move(p2p, old_p2p)
            print(f"Moved to {old_p2p}")

        # Create project folder and subfolders
        self.path2project.mkdir()

        # ########################
        # Add files and subfolders

        # Subfolders
        self._update_folders(None)

        # Place a copy of a default configuration file in the project folder.
        # This file should be adapted by the user to the new project settings.
        p2default_c = pathlib.Path(
            'config', p2c.stem + '.default' + p2c.suffix)
        if p2default_c.is_file():
            # If a default configuration file exists, it is copied into the
            # project folder.
            shutil.copyfile(p2default_c, p2c)

        # #####################
        # Update project status

        # Update the state of the project.
        self.state['isProject'] = True
        self.metadata.update({'state': self.state})

        # Save metadata
        self._save_metadata()
        # The project is ready to setup, but the user should edit the
        # configuration file first
        self.ready2setup = True

        print(f"-- Project {p2p} created.")
        print("---- Project metadata saved in {0}".format(self.metadata_fname))
        print("---- A default config file has been located in the project " +
              "folder.")
        print("---- Open it and set your configuration variables properly.")
        #print("---- Once the config file is ready, activate it.")
        
        # #####################
        # Activate configuration file
        self.setup()

        return

    def load(self):
        """
        Loads an existing project, by reading the metadata file in the project
        folder.

        It can be used to modify file or folder names, or paths, by specifying
        the new names/paths in the f_struct dictionary.
        """

        # ########################
        # Load an existing project
        print("\n*** LOADING PROJECT")

        # Check and clean project folder location
        if not self.path2metadata.exists():
            exit(f'-- ERROR: Metadata file {self.path2metadata} does not' +
                 '   exist.\n' +
                 '   This is likely not a project folder. Select another ' +
                 'project or create a new one.')

        else:
            # Load project metadata
            self.metadata = self._load_metadata()

            # Store state
            self.state = self.metadata['state']

            # The following is used to automatically update any changes in the
            # keys of the self.f_struct dictionary. This will be likely
            # unnecesary once a stable version of the code is reached, but it
            # is useful to update older application projects.
            self._update_folders(self.f_struct)

            if self.state['configReady']:
                self.ready2setup = True
                self.setup()
                print(f'-- Project {self.path2project} succesfully loaded.')
            else:
                exit(f'-- WARNING: Project {self.path2project} loaded, but ' +
                     'configuration file could not be activated. You can: \n' +
                     '(1) revise and reactivate the configuration file, or\n' +
                     '(2) delete the project folder to restart')

        return

