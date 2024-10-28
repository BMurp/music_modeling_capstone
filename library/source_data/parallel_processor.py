from multiprocessing import Process
import sys
sys.path.insert(0, '../../')
from library.source_data.feature_extractor import AudioFeatureExtractor
print('test')

class AudioParallelProcessor():
    '''allows for parallel extraction , for a given number threads source data is divided into 
    chunks and python's multiprocessing.Process is used to start multiple parallel processes
    For each process a AudioFeatureExtractor instance is created for a subset of the data 
    and run_extraction_thread method is called for each in prallel which extracts features to memory 
    and puts result to parque files for each thread. 
    
    '''
    def __init__(self, source_data, version = '003', batch=1, threads = 5):
        self.version = version 
        self.batch = batch
        self.threads = threads
        self.source_data = source_data
        self.input_length = len(self.source_data)
        self.start_thread_size = int(self.input_length/self.threads)
        self.extract_processes ={}

        return

    
    def execute(self):
        '''execution of the processes by populating extract processes with a map of thread index to process
        and then running processes
        '''
        self.get_extract_processes()
        self.run_processes()
    
    def get_extract_processes(self):
        '''populates the extract_threads_extractors method with the run_extraction method 
        of an AudioFeatureExtractor instantiated with a subset of the data 
        '''
        #map of index number and the function to run 
        extract_thread_extractors = {}
        #map of index number and process to execture 
        #extract_processes = {}
        #1 based indexes for the threads 
        thread_indexes = list(range(1,self.threads+1))
   
        #Instantiate the index locations for getting rows from data frame 
        start_record = 0
        end_record = self.start_thread_size 

        #for reach thread build out the processes for a subset of the data 
        for thread in thread_indexes:
            if thread == 2:
                start_record +=1
            if end_record == self.input_length - self.input_length%self.threads:
                end_record += self.input_length%self.threads
            print(f"populate thread process for thread {thread} " ,start_record, end_record, len(self.source_data.iloc[start_record:end_record]))
            extract_thread_extractors[thread] = AudioFeatureExtractor(self.source_data.iloc[start_record:end_record])
            
            #for debugging of just printing out the function calls arguments 
            #self.extract_processes[thread] = Process(target=extract_thread_extractors[thread].logger_function,args=(thread, start_record, end_record))
            self.extract_processes[thread] = Process(target =extract_thread_extractors[thread].run_extraction_thread,args=(self.version,self.batch,thread))
            start_record += self.start_thread_size
            end_record += self.start_thread_size 
            
        return


    def run_processes(self):
        '''invoke start on each process and then join on each process
        this kicks off each process and then instructs python to wait to move forward until all are done
        '''
        for thread_index, process in self.extract_processes.items():
            print(f"start process thread {thread_index}")
            process.start()
        for thread_index, process in self.extract_processes.items():
            process.join()