class Extractor():
    def __init__(self,model,layer):
        pass
    def extract(self,input,unit):
        pass
    def get_class(self):
        return "Extractor"
    
# goole cloud extractor 
# def Build_gcs_extractor(transition_table, dictionary, init_state = 0, id_fsm = None):
#     @ray.remote
#     def F(seq_raw):
#         int2char = {0:'X', 1:'0', 2:'1', 3:'|', 4:'&'}
#         seq = ''
#         #latter not 0
#         for i in seq_raw[0]:
#             for k in range(len(i)):
#                 if i[k] == 1:
#                     seq = seq + int2char[k]
#         features = np.zeros(len(seq))
#         cur_state = init_state
#         for i, x in enumerate(seq):
#             cur_state = transition_table[cur_state][dictionary[x]]
#             features[i] = cur_state
#         name = 'states' if id_fsm is None else 'states_' + id_fsm
#         return features, name
#     return F
