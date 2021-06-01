"""
neurips_arc_experiments.py | Author : Catherine Wong.

This generates the command line output necessary to launch experiments for the LARC domain.

Usage derives from the experiment_utils class.
"""
from experiment_utils import *

ARC_DOMAIN_PREFIX = "arc"

ARC_CHECKPOINT_REGISTRY = {
"local_unigram_enumeration" : "experimentOutputs/arc/2021-04-20T23:35:15.171726/arc_aic=1.0_arity=0_ET=10_t_zero=28800_it=1_MF=10_noConsolidation=True_pc=1.0_RW=False_solver=ocaml_STM=True_L=1.0_TRR=default_K=2_topkNotMAP=False_rec=False_graph=True.pickle",

"om_unigram_enumeration" : "/om2/user/theosech/arc_dc/ec/experimentOutputs/arc/2021-04-20T23:35:15.171726/arc_aic=1.0_arity=0_ET=10_t_zero=28800_it=1_MF=10_noConsolidation=True_pc=1.0_RW=False_solver=ocaml_STM=True_L=1.0_TRR=default_K=2_topkNotMAP=False_rec=False_graph=True.pickle",

'om_bigram_enumeration' : "/om2/user/theosech/arc_dc/ec/experimentOutputs/arc/2021-04-21T20:11:03.040547/arc_arity=0_BO=False_CO=True_ES=1_ET=600_t_zero=3600_HR=0_it=2_MF=10_noConsolidation=True_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_TRR=default_K=2_topkNotMAP=False_DSL=False.pickle"
}

@register_experiment("language_lstm_bigram_only")
def build_experiment_language_lstm_bigram_only(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with no compression. This does not run a unigram enumeration run. 
    """
    def experiment_parameters_fn():
        return  "--no-dsl --recognition_0 --recognition_1 language --Helmholtz 0 --no-consolidation --languageDataset human "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_lstm_no_compression")
def build_experiment_language_lstm_no_compression(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with no compression.
    """
    def experiment_parameters_fn():
        return  " --recognition_0 --recognition_1 language --Helmholtz 0 --no-consolidation --languageDataset human "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_lstm_compression")
def build_experiment_language_lstm_compression(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with compression.
    """
    def experiment_parameters_fn():
        return  " --recognition_0 --recognition_1 language --Helmholtz 0 --languageDataset human "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("unigram_enumeration")
def build_experiment_unigram_enumeration(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with compression.
    """
    def experiment_parameters_fn():
        return  " --featureExtractor dummy --Helmholtz 0 --no-recognition --no-consolidation --unigramEnumerationTimeout 720   "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("dummy_compression_unigram_enum")
def build_experiment_dummy_compression_unigram_enum(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with compression.
    """
    def experiment_parameters_fn():
        return  " --featureExtractor dummy --Helmholtz 0  "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)
    
@register_experiment("language_t5_compression_unigram_enum")
def build_experiment_language_lstm_compression(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with compression.
    """
    def experiment_parameters_fn():
        return  " --featureExtractor LMFeatureExtractor --Helmholtz 0  "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)



@register_experiment("dummy_no_compression")
def build_experiment_dummy_no_compression(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --featureExtractor dummy --Helmholtz 0 --no-dsl --no-consolidation "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("dummy_compression")
def build_experiment_dummy_compression(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with compression.
    """
    def experiment_parameters_fn():
        return  " --featureExtractor dummy --Helmholtz 0 --no-dsl "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("arccnn_no_compression")
def build_experiment_arccnn_no_compression(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with compression.
    """
    def experiment_parameters_fn():
        return  " --featureExtractor arcCNN --Helmholtz 0 --no-dsl --no-consolidation "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)
    
@register_experiment("arccnn_compression")
def build_experiment_arccnn_compression(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with compression.
    """
    def experiment_parameters_fn():
        return  " --featureExtractor arcCNN --Helmholtz 0 --no-dsl "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_t5_no_compression")
def build_experiment_language_lstm_compression(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with compression.
    """
    def experiment_parameters_fn():
        return  " --featureExtractor LMFeatureExtractor --Helmholtz 0 --no-dsl --no-consolidation "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)
    
@register_experiment("language_t5_compression")
def build_experiment_language_lstm_compression(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations and no examples, with compression.
    """
    def experiment_parameters_fn():
        return  " --featureExtractor LMFeatureExtractor --Helmholtz 0 --no-dsl "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_t5_pseudo_translations_no_compression")
def build_experiment_language_t5_pseudo_translations_no_compression(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --featureExtractor LMPseudoTranslationFeatureExtractor --Helmholtz 0.5 --no-background-helmholtz --no-dsl  --no-consolidation"
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_t5_pseudo_translations_compression")
def build_experiment_language_t5_pseudo_translations_compression(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --featureExtractor LMPseudoTranslationFeatureExtractor --Helmholtz 0.5 --no-dsl --no-background-helmholtz "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_t5_cnn_no_compression")
def build_experiment_language_t5_cnn_no_compression(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --featureExtractor LMCNNFeatureExtractor  --Helmholtz 0 --no-background-helmholtz --no-dsl  --no-consolidation"
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_t5_cnn_compression")
def build_experiment_language_t5_cnn_compression(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --featureExtractor LMCNNFeatureExtractor  --Helmholtz 0 --no-dsl --no-background-helmholtz "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_t5_cnn_pseudo_no_compression")
def build_experiment_language_t5_cnn_pseudo_no_compressionn(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --featureExtractor LMCNNPseudoFeatureExtractor --Helmholtz 0 --no-background-helmholtz --no-dsl  --no-consolidation"
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_t5_cnn_pseudo_compression")
def build_experiment_language_t5_cnn_pseudo_compression(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --featureExtractor LMCNNPseudoFeatureExtractor  --Helmholtz 0 --no-dsl --no-background-helmholtz "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_t5_augmented_no_compression")
def build_experiment_language_t5_augmented_no_compression(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --featureExtractor LMAugmentedFeatureExtractor --Helmholtz 0 --no-background-helmholtz --no-dsl  --no-consolidation"
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

@register_experiment("language_t5_augmented_compression")
def build_experiment_language_t5_augmented_compression(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --featureExtractor LMAugmentedFeatureExtractor  --Helmholtz 0 --no-dsl --no-background-helmholtz "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)
    
    
    
def main():
    parser = get_experiment_argparser(domain_name_prefix=ARC_DOMAIN_PREFIX)
    args = parser.parse_args()
    generate_all_launch_commands_and_log_lines(args, checkpoint_registry=ARC_CHECKPOINT_REGISTRY)
    
if __name__ == '__main__':
  main() 
