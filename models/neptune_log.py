import neptune.new as neptune
import os

NEPTUNE_PROJECT = os.getenv('NEPTUNE_PROJECT')
API_TOKEN = os.getenv('NEPTUNE_API_TOKEN')

run = neptune.init(
    project=NEPTUNE_PROJECT,
    api_token=API_TOKEN,
    source_files=['*.py'],
    run='FLOW-286'
)

run['parameters/model'] = 'dwt_horiz_unet'