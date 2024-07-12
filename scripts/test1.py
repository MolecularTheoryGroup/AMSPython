#!/Applications/AMS2024.102.app/Contents/Resources/amshome/bin/amspython

from scm.plams import *

run_file_path = '/Users/haiiro/scratch/h2_test.run'
input_job = AMSJob.from_inputfile(run_file_path)

# input_job.settings['input']['ams']['system']['electrostaticembedding']['electricfield'] = "0.0 0.0 0.0"

s = Settings()
s.electrostaticembedding.electricfield = "0.0 0.0 0.0"

input_job.settings.input.ams.system = s

print(input_job.settings)
# print(s)
print(input_job.get_input())