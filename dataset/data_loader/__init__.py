# Import only the available data loaders
import dataset.data_loader.BaseLoader
import dataset.data_loader.COHFACELoader
import dataset.data_loader.AIRLoader

# Note: The following loaders are referenced in main.py but not yet implemented:
# - UBFCLoader (needed for UBFC dataset)
# - PURELoader (needed for PURE dataset)
# - SCAMPSLoader (needed for SCAMPS dataset)
# - ACLLoader (needed for ACL dataset)
# If you need these datasets, you'll need to add the corresponding loader files.
