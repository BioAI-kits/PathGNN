import torch
import tarfile
import gzip
import glob
import os
import sys
import mygene
import configparser
import numpy as np
import pandas as pd
from torch_geometric.data import Data

# from baseline import F


###############################################################################
#                                                                             #
#                              get configures                                 #
#                                                                             #
###############################################################################
def get_config(config_file = 'LUAD.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    print('[INFO] Perform {} Project.\n'.format(config['DEFAULT']['project']))
    return config


###############################################################################
#                                                                             #
#                        Preprocess data from TCGA                            #
#                                                                             #
###############################################################################
class Preprocess_tcga():
    """
    There are three file in root_dir: 
        (1) clinical.tar.gz
        (2) gdc.tar.gz
        (3) gdc_sample_sheet.tsv
    
    Those files should be download from TCGA websit, and change filename.

    About run:
        First, perform function 'preprocess_expression()', like this:
            Preprocess_tcga(root_dir='/home/PJLAB/liangbilin/Projects/DeepSurvial/Data/GBM_test').preprocess_expression()
        Then, function will output a clean data in clean dir.
    """
    def __init__(self, root_dir, gmt):
        """
        root_dir: root 
        """
        self.root_dir = root_dir
        self.gmt = gmt
    
    
    def check_files(self):
        print('check files.\n')
        # check clinical file
        cli_file = os.path.join(self.root_dir, 'clinical.tar.gz')
        if not os.path.exists(cli_file):
            print('cinical file was not found in: ', cli_file)
            sys.exit(0)
        
        # check expression file
        expr_file = os.path.join(self.root_dir, 'gdc.tar.gz')
        if not os.path.exists(expr_file):
            print('Gene expression file was not found in: ', expr_file)
            sys.exit(0)

        # sample information file
        smp_file = os.path.join(self.root_dir, 'gdc_sample_sheet.tsv')
        if not os.path.exists(smp_file):
            print('sample information file was not found in: ', smp_file)
            sys.exit(0)
        return cli_file, expr_file, smp_file


    def un_gz(self, file_name):
        """ungz zip file"""
        f_name = file_name.replace(".gz", "")
        #获取文件的名称，去掉
        g_file = gzip.GzipFile(file_name)
        #创建gzip对象
        open(f_name, "wb+").write(g_file.read())
        #gzip对象用read()打开后，写入open()建立的文件里。
        g_file.close()
        #关闭gzip对象


    def uncompress(self):
        print('uncompress files.\n')
        cli_file, expr_file, smp_file = self.check_files()
        
        # uncompress clinical data
        t = tarfile.open(cli_file)
        t.extractall(path = self.root_dir)

        # uncompress gene expression data
        try:
            os.mkdir(os.path.join(self.root_dir, 'Expression'))
        except:
            pass
        t = tarfile.open(expr_file)
        t.extractall(path = os.path.join(self.root_dir, 'Expression'))
        
        # uncompress gene expression per sample
        os.path.join(self.root_dir, 'Expression', "*/*gz")
        files = glob.glob(os.path.join(self.root_dir, 'Expression', "*/*gz"))
        for f in files:
            self.un_gz(file_name=f)
            
        return cli_file, expr_file, smp_file

    def preprocess_cli(self):
        print('preprocess clinical files.\n')
        ##### run uncompress  #####
        cli_file, expr_file, smp_file = self.uncompress()
        
        ##### preprocess clinical data  #####
        # extract columns
        tb_cli = pd.read_table(os.path.join(self.root_dir, 'clinical.tsv'))
        df_cli = tb_cli.loc[:,['case_submitter_id', 'vital_status', 'days_to_last_follow_up', 'days_to_death','age_at_diagnosis', 'gender','ajcc_pathologic_stage', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'ajcc_pathologic_t']]
        
        # transform os time
        for idx, row in df_cli.iterrows():
            if row[3] == "'--":
                df_cli.loc[idx, 'days_to_death'] = row[2]
                
        # transform survival status
        def f1(x):
            if x == "Dead":
                return 1
            elif x == "Alive":
                return 0
            else :
                return 'na'
        df_cli.vital_status = df_cli.vital_status.map(f1)
        
        # drop duplicated case
        df_cli = df_cli.drop_duplicates('case_submitter_id')
        
        # rename case id
        df_cli = df_cli.rename(columns={'case_submitter_id': 'Case ID'})
        
        ##### preprocess smaple file ####
        tb_smp = pd.read_table(smp_file)
        tb_smp = tb_smp.loc[:, ['File ID', 'File Name', 'Case ID', 'Sample Type']]
        
        ##### merge clinical and smaple file ####
        df_cli = pd.merge(left=df_cli, right=tb_smp, on='Case ID', how='left').drop_duplicates('Case ID')
        
        ##### output file1: Patient_ID | Status | Time  #####
        try:
            os.mkdir(os.path.join(self.root_dir, 'clean'))
        except:
            pass
        cli_out = df_cli.loc[:, ['Case ID', 'vital_status', 'days_to_death', 'age_at_diagnosis','gender','ajcc_pathologic_stage', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'ajcc_pathologic_t']]
        
        # add step for clinical features
        def f2(x):
            x = x.strip()
            if x in ['Stage IA', 'Stage IB', 'Stage I']:
                return 0.2
            elif x in ['Stage IIA', 'Stage IIB', 'Stage II', 'Stage IIC']:
                return 0.4
            elif x in ['Stage IIIA', 'Stage IIIB', 'Stage III', 'Stage IIIC']:
                return 0.6
            elif x in ['Stage IVA', 'Stage IVB', 'Stage IV', 'Stage IVC']:
                return 0.8
            elif x in ["'--", 'Stage 0', 'Not Reported']:
                return 0
            else:
                print('[Error] in clinical feature, Stage.', x, '/n')
                sys.exit(1)
        cli_out['ajcc_pathologic_stage_score'] = cli_out.ajcc_pathologic_stage.map(f2)
        def f3(x):
            x = str(x).strip()
            try:
                x = int(x)
                return x / 36500
            except:
                return 0.5
        cli_out['age_norm'] = cli_out['age_at_diagnosis'].map(f3)

        cli_out.columns = ['Patient_ID','Status','Time', 'age','gender','ajcc_pathologic_stage', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'ajcc_pathologic_t', 'ajcc_pathologic_stage_score', 'age_norm']
        cli_out = cli_out[(cli_out.Status != 'na') &(cli_out.Time !=  "'--")]
        cli_out.to_csv(os.path.join(self.root_dir, 'clean/clinical.txt'), index=False, sep='\t')
        
        
        ##### output file2: Case ID | File ID | File Name | path_  #####
        cli_file = df_cli.loc[:, ['Case ID', 'File ID', 'File Name']]
        cli_file = cli_file[cli_file['Case ID'].isin(cli_out.Patient_ID)]
        for idx, row in cli_file.iterrows():
            cli_file.loc[idx, 'path_'] = row[1] + '/' + row[2][:-3]
        cli_file.to_csv(os.path.join(self.root_dir, 'clean/expression_file_path.txt'), index=False, sep='\t')
        
        return cli_out, cli_file
        
  
    def preprocess_expression(self):
        print('preprocess gene expression files.\n')
        df_survival, cli_file = self.preprocess_cli()
        dfs = []
        for idx, row in cli_file.iterrows():
            patient_id = row[0]
            path_ = os.path.join(self.root_dir, 'Expression', row[3])
            df_expr = pd.read_table(path_, header=None)
            df_expr.columns = ['ENSG', patient_id]
            dfs.append(df_expr)
        df = pd.concat(dfs, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        def f(x):
            return x.split('.')[0]
        df.ENSG = df.ENSG.map(f)

        mg = mygene.MyGeneInfo()
        df_change = mg.querymany(df.ENSG.to_list(), 
                                 scopes='ensembl.gene', 
                                 fields='entrezgene', 
                                 species='human', 
                                 as_dataframe=True)
        
        def f1(x):
            return df_change.loc[x, 'entrezgene']
        df['ENSG'] = df['ENSG'].map(f1)
        df = df[~df['ENSG'].isnull()]
        df = df.rename(columns={'ENSG': 'Entrez_Gene_Id'})

        # filter bad line
        bad_idx = []
        for idx, row in df.iterrows():
            eid = row[0]
            try:
                int(eid)
            except:
                bad_idx.append(idx)
        df = df.drop(bad_idx).reset_index(drop=True)
        df = df.drop_duplicates('Entrez_Gene_Id')

        df.to_csv(os.path.join(self.root_dir, 'clean/expression_matrix.txt.old'), index=False, sep='\t')
        # only keep pathway relative genes (update @2021-11-03)
        genes = []
        with open(self.gmt) as F:
            lines = [line.strip() for line in F.readlines()]
            for line in lines:
                genes += line.split('\t')[1:]
        genes = list(set(genes))
        df = df[df['Entrez_Gene_Id'].isin(genes)]

        # output
        df.to_csv(os.path.join(self.root_dir, 'clean/expression_matrix.txt'), index=False, sep='\t')
        
        
###############################################################################
#                                                                             #
#                      Preprocess pathway from Reactome                       #
#                                                                             #
###############################################################################
class Preprocess_pathway():
    """
    Used gene expression and pathway network to construct graph dataset.

    Args:
        (1) gene_expression_file: Gene expression matrix file from 'Preprocess_tcga' class or sampe format.
            the format should be like this (sep=TAB):
                Entrez_Gene_Id | sample1 | sample2 | sample3 | ...

        (2) pathway_files: pathway network file
            the network file should include two columns named: "scr": sorce node; "dest": dest node

        (3) save_dataset: set graph dataset saved filename.
    """
    def __init__(self, gene_expression_file='/home/PJLAB/liangbilin/Projects/DeepSurvial/Data/GBM/clean/expression_matrix.txt',
                       pathway_files='/home/PJLAB/liangbilin/Projects/muti-omics/pathways/R*.txt',
                       save_dataset ='/home/PJLAB/liangbilin/Projects/DeepSurvial/Data/GBM/clean/pathway_reactome.pt',
                       keep_pathway='/home/PJLAB/liangbilin/Projects/DeepSurvial/Pathway/keep_pathways_details.tsv',
                       pathway_score=None,
                       ):
        self.gene_expression_file = gene_expression_file
        self.pathway_files = pathway_files
        self.save_dataset = save_dataset
        self.pathway_score = pathway_score
        self.keep_pathway = keep_pathway

    def get_score(self, smp, pathway_name):
        """
        get pathway activate level from ssGSEA.

        smp: patient id
        pathway_name: pathway_name
        """
        if self.pathway_score != None:
            df_score = pd.read_csv(self.pathway_score)
            df_score.columns = ['Patient_ID'] + [i.replace('.', '-') for i in df_score.columns[1:]]
            df_score = df_score.set_index('Patient_ID')
            df_score = df_score.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            try:
                return torch.tensor(round(df_score.loc[pathway_name, smp], 3))
            except:
                return None
        else:
            return None


    def gene2graph(self, df, df_expr, smp, pathway_name):
        """
        construct graph

        df: pathway network info, dataframe.
        df_expr: gene expression info, dataframe.
        smp: patient id
        pathway_name: pathway_name
        """
        # 按照Gene名（entrez_id)升序
        genes = df.src.to_list()
        genes = list(set(genes + df.dest.to_list()))
        genes = sorted(genes)
        
        ######### node feature #########
        # extract row with expression level
        df_tmp = df_expr[df_expr.Entrez_Gene_Id.isin(genes)].loc[:, ['Entrez_Gene_Id', smp]]
        
        # extract gene with expression level
        genes_tmp = df_tmp.Entrez_Gene_Id.to_list()

        # format gene without expression level
        df_1 = pd.DataFrame([list(set(genes).difference(set(genes_tmp))),
                    [df_tmp.iloc[:, 1].mean()]*len(list(set(genes).difference(set(genes_tmp))))
                    ]).T
        df_1.columns = df_tmp.columns

        df_2 = pd.concat([df_tmp, df_1], axis=0, ignore_index=True)
        df_2 = df_2.drop_duplicates('Entrez_Gene_Id', keep='first').sort_values('Entrez_Gene_Id')
        
        node_features = torch.tensor(df_2.loc[:, smp].values.reshape(-1,1))

        # 序号作为gene的图节点
        edge_index = np.array([[genes.index(i) for i in df.src.values],
                            [genes.index(i) for i in df.dest.values]])  
        edge_index = torch.tensor(np.unique(edge_index, axis=1),dtype=torch.long) # 删除重复边

        # 构建图
        if self.pathway_score == None:
            data = Data(edge_index=edge_index, x=node_features.to(torch.float32))
        else:
            score_ = self.get_score(smp=smp, pathway_name=pathway_name)
            data = Data(edge_index=edge_index, x=node_features.to(torch.float32), y=score_)

        return data


    def batch_gene2graph(self):
        ####  read gene expression file: Entrez_Gene_Id | sample_id | sample_id | ...
        df_expr = pd.read_table(self.gene_expression_file)
        df_expr = df_expr[~df_expr.Entrez_Gene_Id.isnull()]  # remove NULL
        df_expr.Entrez_Gene_Id = df_expr.Entrez_Gene_Id.astype(int)  # change Enrez Gene Name to int type
        df_expr = df_expr.sort_values('Entrez_Gene_Id')
        df_expr.iloc[:,1:] = (df_expr.iloc[:,1:] - df_expr.iloc[:,1:].min()) / (df_expr.iloc[:,1:].max() - df_expr.iloc[:,1:].min())    # change to 0-1
        # df_expr.iloc[:,1:] = (df_expr.iloc[:,1:] - df_expr.iloc[:,1:].mean()) / (df_expr.iloc[:,1:].std())    # change to z-score

        #### constrcut graph
        # dict: for save train data: sample  --> pathway
        dataset = {}  # 所有样本的图数据
                
        i = 0
        for smp in df_expr.columns[1:]:
            print('running {}:'.format(i), smp)
            pathway_names = []  #存储pathway名
            i += 1

            dat = []  # 单个患者的所有pathway图， list类型
            files = glob.glob(self.pathway_files)
            keep_pathways = pd.read_table(self.keep_pathway)['Native ID'].to_list()
            for file in files:
                pathway_name = file.split('/')[-1].split('.')[0]
                if pathway_name not in keep_pathways:
                    # print('{} not in keep pathways.'.format(pathway_name))
                    continue
                df = pd.read_table(file)
                df = df[df.direction == 'directed']  ## 基因直接相互作用：在Graph中才有Edge
                data = self.gene2graph(df, df_expr=df_expr, smp=smp, pathway_name=pathway_name)
                
                # 去除节点小于15的图
                # if df.shape[0] < 15:
                #     continue
                # if data.num_nodes < 15:
                #     continue
                    
                dat.append(data)
                pathway_names.append(pathway_name)
        
            dataset[smp] = dat

        # save dataset
        torch.save(dataset, self.save_dataset)
        # output pathway name
        if os.path.exists(self.save_dataset + '.pathway_names.txt'):
            os.remove(self.save_dataset + '.pathway_names.txt')
        with open(self.save_dataset + '.pathway_names.txt', 'a') as F:
            for line in pathway_name:
                F.writelines(line + '\n')


###############################################################################
#                                                                             #
#                        Change pathway to gmt format                         #
#                                                                             #
###############################################################################
def generate_gmt(dtail_file, save_file):
    """
    get gmt format pathway file for ssgsea.
    
    dtail_file: keep_pathways_details.tsv
    save_file: gmt pathway file
    """
    df_dtails = pd.read_table(dtail_file)

    # remove old file
    with open(save_file, 'w') as F:
        F.writelines('')

    for idx, row in df_dtails.iterrows():
        id_ = row[1]
        df = pd.read_table('Pathway/pathways/{}.txt'.format(id_))
        genes = df.src.to_list()
        genes = list(set(genes + df.dest.to_list()))
        genes = sorted(genes)
        num_genes = len(genes)
        df_dtails.loc[idx, 'Number_of_nodes_update'] = int(num_genes)
        # output
        line = '\t'.join([id_] + [str(g) for g in genes])
        with open(save_file, 'a') as F:
            F.writelines(line + '\n')

    df_dtails.to_csv(dtail_file, index=False, sep='\t')


###############################################################################
#                                                                             #
#                              Main function                                  #
#                                                                             #
###############################################################################
if __name__ == '__main__':
    # read configures
    config = get_config(sys.argv[1])
    project_root_dir = config['DEFAULT']['root_dir']
    data_dir = config['DATA']['data_dir']

    # step1: get gmt pathway file
    generate_gmt(dtail_file=os.path.join(project_root_dir, config['DATA']['keep_pathway']), 
                 save_file=os.path.join(project_root_dir, config['DATA']['pathway_gmt']))

    # step2: preprocess data
    Preprocess_tcga(root_dir=os.path.join(project_root_dir, data_dir),
                    gmt=os.path.join(project_root_dir, config['DATA']['pathway_gmt'])
    ).preprocess_expression()

      
    # step3: calculate ssgsea
    os.system('Rscript ssgsea.R {} {} {}'.format(os.path.join(project_root_dir, data_dir, 'clean/expression_matrix.txt'),
                                                 os.path.join(project_root_dir, config['DATA']['pathway_gmt']),
                                                 os.path.join(project_root_dir, data_dir, 'expression_matrix.ssgsea.csv')

            ))

    # step4: Preprocess_pathway
    Preprocess_pathway(gene_expression_file=os.path.join(project_root_dir, data_dir, 'clean/expression_matrix.txt'),
                       pathway_files=os.path.join(project_root_dir, config['DATA']['pathway_dir']),
                       pathway_score=os.path.join(project_root_dir, data_dir, 'expression_matrix.ssgsea.csv'),
                       save_dataset =os.path.join(project_root_dir, data_dir, 'clean/pathway_reactome_ssgsea.pt'),
                       keep_pathway = os.path.join(project_root_dir, config['DATA']['keep_pathway'])
    ).batch_gene2graph()

