def check_agisoft_license():
    licensed : bool = False

    try:
        import Metashape
        installed = True

        if Metashape.license.valid:
            licensed = True
        else:
            licensed = False

    except ModuleNotFoundError:
        installed = False
        licensed = False

    except ImportError:
        licensed = False
        installed = False

    return installed, licensed

