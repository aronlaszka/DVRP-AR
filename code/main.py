import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # this configuration to work with non-lab servers
    from learn.settings import set_threads

    set_threads(number_of_threads=1, log_changes=True)

    from env.Coordinator import Coordinator
    Coordinator().update_configurations().run()
