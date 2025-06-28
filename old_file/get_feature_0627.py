DNA2Protein = {
        'TTT' : 'F', 'TCT' : 'S', 'TAT' : 'Y', 'TGT' : 'C',
        'TTC' : 'F', 'TCC' : 'S', 'TAC' : 'Y', 'TGC' : 'C',
        'TTA' : 'L', 'TCA' : 'S', 'TAA' : '*', 'TGA' : '*',
        'TTG' : 'L', 'TCG' : 'S', 'TAG' : '*', 'TGG' : 'W',

        'CTT' : 'L', 'CCT' : 'P', 'CAT' : 'H', 'CGT' : 'R',
        'CTC' : 'L', 'CCC' : 'P', 'CAC' : 'H', 'CGC' : 'R',
        'CTA' : 'L', 'CCA' : 'P', 'CAA' : 'Q', 'CGA' : 'R',
        'CTG' : 'L', 'CCG' : 'P', 'CAG' : 'Q', 'CGG' : 'R',

        'ATT' : 'I', 'ACT' : 'T', 'AAT' : 'N', 'AGT' : 'S',
        'ATC' : 'I', 'ACC' : 'T', 'AAC' : 'N', 'AGC' : 'S',
        'ATA' : 'I', 'ACA' : 'T', 'AAA' : 'K', 'AGA' : 'R',
        'ATG' : 'M', 'ACG' : 'T', 'AAG' : 'K', 'AGG' : 'R',

        'GTT' : 'V', 'GCT' : 'A', 'GAT' : 'D', 'GGT' : 'G',
        'GTC' : 'V', 'GCC' : 'A', 'GAC' : 'D', 'GGC' : 'G',
        'GTA' : 'V', 'GCA' : 'A', 'GAA' : 'E', 'GGA' : 'G',
        'GTG' : 'V', 'GCG' : 'A', 'GAG' : 'E', 'GGG' : 'G',
        'nnn':'n'
}

# >A,>T,>G,>Cの更新は無し
def Feature_from_csv(mutation, codon_df, bunpu_df):

    base_pos = int(mutation[1:-1])
    bef = str(mutation[0])
    aft = str(mutation[-1])

    base = str(codon_df["base"][base_pos-1])
    protein = str(codon_df["protein"][base_pos-1])
    protein_pos = int(codon_df["protein_pos"][base_pos-1])
    codon = str(codon_df["codon"][base_pos-1])
    codon_pos = int(codon_df["codon_pos"][base_pos-1])
    #print(f"base_pos: {base_pos}, bef: {bef}, aft: {aft}, base: {base}, protein: {protein}, protein_pos: {protein_pos}, codon: {codon}, codon_pos: {codon_pos}")

    if bef == base and codon != "none":
        codon_df.at[base_pos-1, "base"] = aft
        if codon_pos == 1:
            new_codon = aft + codon[1:3]
            base1, base2 = 1, 2
        elif codon_pos == 2:
            new_codon = codon[0] + aft + codon[2]
            base1, base2 = -1, 1
        elif codon_pos == 3:
            new_codon = codon[0:2] + aft
            base1, base2 = -2, -1
        else:
            new_codon = codon  # 念のため
            base1 = base2 = 0

        codon_df.at[base_pos-1, "codon"] = new_codon
        if protein_pos == codon_df["protein_pos"][base_pos-1+base1]:
            codon_df.at[base_pos-1+base1, "codon"] = new_codon
        if protein_pos == codon_df["protein_pos"][base_pos-1+base2]:
            codon_df.at[base_pos-1+base2, "codon"] = new_codon
    else:
        new_codon = codon
    
    freq = bunpu_df[bef+'->'+aft][base_pos-1]

    return codon, new_codon, codon_pos, protein, protein_pos,freq

def Mutation_features(mutations, codon_df, bunpu_df):
    codon_str, amino_str, freq_str, protein_str, codon_pos_str = '', '', '', '', ''
    for mutation in mutations.split(','):
        codon, new_codon, codon_pos, protein, protein_pos, freq = Feature_from_csv(mutation, codon_df, bunpu_df)
        if(codon=='none'):
            codon = 'nnn'
        if(new_codon=='none'):
            new_codon = 'nnn'
        codon_str += f"{codon}{protein_pos}{new_codon},"
        amino_str += f"{DNA2Protein[codon]}{protein_pos}{DNA2Protein[new_codon]},"
        freq_str += f"{freq},"
        protein_str += f"{protein},"
        codon_pos_str += f"{codon_pos},"
    return codon_str[:-1], amino_str[:-1], protein_str[:-1], codon_pos_str[:-1], freq_str[:-1]


def Feature_path(mutation_path, codon_df, bunpu_df):
    codon_df1 = codon_df.copy()
    result = {
        'codon': [],
        'amino': [],
        'freq': [],
        'protein': [],
        'codon_pos': []
    }
    for mutations in mutation_path:
        codon_str, amino_str, protein_str, codon_pos_str, freq_str = Mutation_features(mutations, codon_df1, bunpu_df)
        result['codon'].append(codon_str)
        result['amino'].append(amino_str)
        result['freq'].append(freq_str)
        result['protein'].append(protein_str)
        result['codon_pos'].append(codon_pos_str)
    return result['codon'], result['amino'], result['freq'], result['protein'], result['codon_pos']

def Separete_HGVS2(hgvs_path):
    bef_path = []
    aft_path = []
    pos_path = []
    for hgvss in hgvs_path:
        bef, aft, pos = "", "", ""
        for hgvs in hgvss.split(','):
            bef += hgvs[0]+","
            aft += hgvs[-1]+ ","
            pos += hgvs[1:-1]+ ","
        bef_path.append(bef[:-1])
        aft_path.append(aft[:-1])
        pos_path.append(pos[:-1])
    return bef_path, aft_path, pos_path

def Separete_HGVS(hgvs_path):
    base_path = []
    pos_path = []
    for hgvss in hgvs_path:
        base, pos = "", ""
        for hgvs in hgvss.split(','):
            base += hgvs[0]+">"+hgvs[-1]+","
            pos += hgvs[1:-1]+ ","
        base_path.append(base[:-1])
        pos_path.append(pos[:-1])
    return base_path, pos_path

def amino_change_flag2(amino_bef, amino_aft):
    flag_list = []
    for befs, afts in zip(amino_bef, amino_aft):
        flag = ""
        for bef, aft in zip(befs.split(','), afts.split(',')):
            if bef == aft:
                flag+= "syno,"
            else:
                flag+= "non-syno,"
        flag = flag[:-1]  # 最後のカンマを削除
        flag_list.append(flag)
    return flag_list

def amino_change_flag(amino_path):
    flag_list = []
    for aminos in amino_path:
        flag = ""
        for amino in aminos.split(','):
            #print(amino)
            if amino[0] == amino[-1]:
                flag+= "syno,"
            else:
                flag+= "non-syno,"
        flag = flag[:-1]  # 最後のカンマを削除
        flag_list.append(flag)
    return flag_list

class MutationFeature_incl_ts:
    def __init__(self, timestep, base_mut, base_pos, amino_mut, amino_pos, amino_flag, freq, protein, codon_pos):
        self.timestep = timestep
        self.base_mut = base_mut
        self.base_pos = "b_"+base_pos
        self.amino_mut = amino_mut
        self.amino_pos = "a_"+amino_pos
        self.amino_flag = amino_flag
        self.protein = protein
        self.codon_pos = "c_"+codon_pos
        self.freq = int(freq)
    
    def print(self):
        print(f"timestep: {self.timestep}, base_mut: {self.base_mut}, base_pos: {self.base_pos}, "
              f"amino_mut: {self.amino_mut}, amino_pos: {self.amino_pos}, amino_flag: {self.amino_flag}, "
              f"freq: {self.freq}, protein: {self.protein}, codon_pos: {self.codon_pos}")
    
    def to_dict(self):
        return {
            'timestep': self.timestep,
            'base_mut': self.base_mut,
            'base_pos': self.base_pos,
            'amino_mut': self.amino_mut,
            'amino_pos': self.amino_pos,
            'amino_flag': self.amino_flag,
            'protein': self.protein,
            'codon_pos': self.codon_pos,
            'freq': self.freq
        }
    def value(self):
        return [self.timestep, self.base_mut, self.base_pos, self.amino_mut, self.amino_pos, self.amino_flag, self.freq, self.protein, self.codon_pos]

def Feature_path_incl_ts(base_HGVS_paths, codon_df, bunpu_df):
    datas = []
    for i in range(0, len(base_HGVS_paths)):
        codon_path, amino_path, freq_path, protein_path, codon_pos_path = Feature_path(base_HGVS_paths[i], codon_df, bunpu_df)
        base_mutation_path, base_pos_path = Separete_HGVS(base_HGVS_paths[i])
        amino_mutation_path, amino_pos_path = Separete_HGVS(amino_path)
        amino_flag_path = amino_change_flag(amino_mutation_path)


        data_ts = {}
        for i in range(len(base_mutation_path)):
            base_mut = base_mutation_path[i].split(',')
            base_pos = base_pos_path[i].split(',')
            amino_mut = amino_mutation_path[i].split(',')
            amino_pos = amino_pos_path[i].split(',')
            amino_flag = amino_flag_path[i].split(',')
            freq = freq_path[i].split(',')
            protein = protein_path[i].split(',')
            codon_pos = codon_pos_path[i].split(',')
            for bm, bp, am, ap, af, f, p, cp in zip(base_mut, base_pos, amino_mut, amino_pos, amino_flag, freq, protein, codon_pos):
                if bm != '' and bp != '' and am != '' and ap != '' and f != '' and p != '' and cp != '':
                    #data.append(MutationFeature_incl_ts(i+1, bm, bp, am, ap, af, f, p, cp).value())
                    if data_ts.get(i+1) is None:
                        data_ts[i+1] = []
                    data_ts[i+1].append(["ts_"+str(i+1), bm, "b_"+bp, am, "a_"+ap, af, p, "c_"+cp, int(f)])
            
        datas.append(data_ts)
    return datas

class MutationFeature_by_ts:
    def __init__(self, base_mut, base_pos, amino_mut, amino_pos, amino_flag, freq, protein, codon_pos):
        self.base_mut = base_mut
        self.base_pos = "b_"+base_pos
        self.amino_mut = amino_mut
        self.amino_pos = "a_"+amino_pos
        self.amino_flag = amino_flag
        self.protein = protein
        self.codon_pos = "c_"+codon_pos
        self.freq = int(freq)
    
    def print(self):
        print(f"base_mut: {self.base_mut}, base_pos: {self.base_pos}, "
              f"amino_mut: {self.amino_mut}, amino_pos: {self.amino_pos}, amino_flag: {self.amino_flag}, "
              f"freq: {self.freq}, protein: {self.protein}, codon_pos: {self.codon_pos}")
    
    def to_dict(self):
        return {
            'base_mut': self.base_mut,
            'base_pos': self.base_pos,
            'amino_mut': self.amino_mut,
            'amino_pos': self.amino_pos,
            'amino_flag': self.amino_flag,
            'protein': self.protein,
            'codon_pos': self.codon_pos,
            'freq': self.freq
        }

def Feature_path_by_ts(base_HGVS_paths, codon_df, bunpu_df):
    datas = []
    for i in range(0, len(base_HGVS_paths)):
        codon_path, amino_path, freq_path, protein_path, codon_pos_path = Feature_path(base_HGVS_paths[i], codon_df, bunpu_df)
        base_mutation_path, base_pos_path = Separete_HGVS(base_HGVS_paths[i])
        amino_mutation_path, amino_pos_path = Separete_HGVS(amino_path)
        amino_flag_path = amino_change_flag(amino_mutation_path)


        data_ts = {}
        for i in range(len(base_mutation_path)):
            data_ts[i+1] = []
            base_mut = base_mutation_path[i].split(',')
            base_pos = base_pos_path[i].split(',')
            amino_mut = amino_mutation_path[i].split(',')
            amino_pos = amino_pos_path[i].split(',')
            amino_flag = amino_flag_path[i].split(',')
            freq = freq_path[i].split(',')
            protein = protein_path[i].split(',')
            codon_pos = codon_pos_path[i].split(',')
            for bm, bp, am, ap, af, f, p, cp in zip(base_mut, base_pos, amino_mut, amino_pos, amino_flag, freq, protein, codon_pos):
                if bm != '' and bp != '' and am != '' and ap != '' and f != '' and p != '' and cp != '':
                    data_ts[i+1].append(MutationFeature_by_ts(bm, bp, am, ap, af, f, p, cp).to_dict())
        

        datas.append(data_ts)
    return datas

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Tuple, Any

class FeatureProcessor:
    def __init__(self, raw_grouped_data: List[List[List[Any]]]):
        self.vocab = defaultdict(lambda: defaultdict(int))
        
        # Define mappings for the *elements within each individual mutation event list*
        # Output of Feature_path_incl_ts produces:
        # [timestep_id, base_mut, "b_"+base_pos, amino_mut, "a_"+amino_pos, amino_flag, protein, "c_"+codon_pos, freq]
        self.feature_map = {
            1: {"name": "base_mut", "type": "categorical"},
            2: {"name": "b_id", "type": "categorical"},
            3: {"name": "amino_mut", "type": "categorical"},
            4: {"name": "a_id", "type": "categorical"},
            5: {"name": "amino_flag", "type": "categorical"},
            6: {"name": "protein_region", "type": "categorical"},
            7: {"name": "c_id", "type": "categorical"},
            8: {"name": "freq_value", "type": "numerical"}, # 'f' in original, assumed numerical
        }

        # Build vocabulary for categorical features by iterating through the grouped data
        for timestep_events in raw_grouped_data: # Each element is a list of events for one timestep
            for event in timestep_events: # Each element is a single mutation event
                # event[0] is the timestep ID, skip it for feature extraction
                for internal_idx, feat_info in self.feature_map.items():
                    if feat_info["type"] == "categorical":
                        # Access the feature value by its internal index in the event list
                        category_value = event[internal_idx] 
                        if category_value not in self.vocab[feat_info["name"]]:
                            self.vocab[feat_info["name"]][category_value] = len(self.vocab[feat_info["name"]])

    def get_vocab_sizes(self) -> Dict[str, int]:
        return {name: len(vocab) for name, vocab in self.vocab.items()}

    def process_event(self, event: List[Any], max_numerical_val: float = 500.0) -> Dict[str, torch.Tensor]:
        """
        Processes a single mutation event list (e.g., [1, 'C>T', ..., '59'])
        into a dictionary of tensors suitable for MutationEventEncoder.
        """
        processed_dict = {}
        for internal_idx, feat_info in self.feature_map.items():
            val = event[internal_idx] # Access value from the event list
            if feat_info["type"] == "categorical":
                processed_dict[feat_info["name"]] = torch.tensor(
                    self.vocab[feat_info["name"]].get(val, 0), dtype=torch.long
                )
            elif feat_info["type"] == "numerical":
                # Ensure conversion to float
                processed_dict[feat_info["name"]] = torch.tensor(
                    float(val) / max_numerical_val, dtype=torch.float
                )
        return processed_dict

# 2. MutationEventEncoder (No change from previous explanation, as it relies on FeatureProcessor's output)
class MutationEventEncoder(nn.Module):
    def __init__(self, feature_processor: FeatureProcessor, embedding_dim: int, max_numerical_val: float = 500.0):
        super().__init__()
        self.feature_processor = feature_processor
        self.embedding_dim = embedding_dim
        self.max_numerical_val = max_numerical_val
        
        self.embeddings = nn.ModuleDict()
        for name, size in self.feature_processor.get_vocab_sizes().items():
            self.embeddings[name] = nn.Embedding(size, embedding_dim)
        
        num_numerical_features = sum(1 for f in self.feature_processor.feature_map.values() if f['type'] == 'numerical')
        self.numerical_projection = nn.Linear(num_numerical_features, embedding_dim)

        num_categorical_features = sum(1 for f in self.feature_processor.feature_map.values() if f['type'] == 'categorical')
        self.input_mutation_dim = (num_categorical_features * embedding_dim) + (num_numerical_features * embedding_dim)
        
        self.final_projection = nn.Linear(self.input_mutation_dim, embedding_dim)

    def forward(self, event_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedded_features = []
        numerical_values = []
        
        # Collect embedded categorical features and numerical values
        for _, feat_info in self.feature_processor.feature_map.items():
            name = feat_info["name"]
            if feat_info["type"] == "categorical":
                embedded_features.append(self.embeddings[name](event_dict[name]))
            elif feat_info["type"] == "numerical":
                numerical_values.append(event_dict[name].unsqueeze(-1)) # Add dim for concatenation

        # Concatenate numerical values if any
        if numerical_values:
            concatenated_numerical = torch.cat(numerical_values, dim=-1)
            projected_numerical = self.numerical_projection(concatenated_numerical)
            embedded_features.append(projected_numerical)
        
        # Concatenate all embedded features
        concatenated_all_features = torch.cat(embedded_features, dim=-1)
        
        return self.final_projection(concatenated_all_features)


# 3. TimestepSummaryTransformer (No change from previous explanation)
class TimestepSummaryTransformer(nn.Module):
    def __init__(self, feature_processor: FeatureProcessor, embedding_dim: int, num_heads: int, num_layers: int, max_mutations_per_timestep: int = 20, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_mutations_per_timestep = max_mutations_per_timestep
        
        self.mutation_event_encoder = MutationEventEncoder(feature_processor, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        self.positional_encoding = nn.Parameter(torch.randn(1, max_mutations_per_timestep + 1, embedding_dim))

    def forward(self, batch_timesteps_events: List[List[Dict[str, torch.Tensor]]]) -> torch.Tensor:
        batch_summary_vectors = []
        
        for events_in_one_timestep in batch_timesteps_events:
            if not events_in_one_timestep:
                batch_summary_vectors.append(torch.zeros(1, self.embedding_dim, device=self.cls_token.device))
                continue

            encoded_mutations = []
            for event_dict in events_in_one_timestep:
                encoded_mutations.append(self.mutation_event_encoder(event_dict))
            
            mutation_sequence = torch.stack(encoded_mutations).unsqueeze(0)

            cls_token_expanded = self.cls_token.expand(1, -1, -1) 
            
            transformer_input = torch.cat((cls_token_expanded, mutation_sequence), dim=1)
            
            if transformer_input.size(1) > self.max_mutations_per_timestep + 1:
                # Handle cases where a timestep has more mutations than max_mutations_per_timestep
                # Options: Truncate, Pad (and mask), or raise error.
                # For this example, we'll truncate or raise an error for simplicity.
                # A robust solution might involve dynamic sequence lengths and masking.
                print(f"Warning: Timestep has {mutation_sequence.size(1)} mutations, exceeding max_mutations_per_timestep ({self.max_mutations_per_timestep}). Truncating sequence.")
                transformer_input = transformer_input[:, :(self.max_mutations_per_timestep + 1), :]
            
            # Apply positional encoding
            transformer_input_with_pos = transformer_input + self.positional_encoding[:, :transformer_input.size(1), :]

            transformer_output = self.transformer_encoder(transformer_input_with_pos)
            
            summary_vector = transformer_output[:, 0, :] 
            batch_summary_vectors.append(summary_vector)

        return torch.cat(batch_summary_vectors, dim=0)