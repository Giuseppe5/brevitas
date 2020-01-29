Search.setIndex({docnames:["brevitas","brevitas.core","brevitas.function","brevitas.loss","brevitas.nn","brevitas.proxy","brevitas.utils","index","modules"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,sphinx:56},filenames:["brevitas.rst","brevitas.core.rst","brevitas.function.rst","brevitas.loss.rst","brevitas.nn.rst","brevitas.proxy.rst","brevitas.utils.rst","index.rst","modules.rst"],objects:{"":{brevitas:[0,0,0,"-"]},"brevitas.config":{env_to_bool:[0,1,1,""]},"brevitas.core":{bit_width:[1,0,0,"-"],function_wrapper:[1,0,0,"-"],quant:[1,0,0,"-"],restrict_val:[1,0,0,"-"],scaling:[1,0,0,"-"],stats:[1,0,0,"-"]},"brevitas.core.bit_width":{BitWidthConst:[1,2,1,""],BitWidthImplType:[1,2,1,""],BitWidthParameter:[1,2,1,""],LsbTruncParameterBitWidth:[1,2,1,""],MsbClampParameterBitWidth:[1,2,1,""],RemoveBitwidthParameter:[1,2,1,""],ZeroLsbTruncBitWidth:[1,2,1,""]},"brevitas.core.bit_width.BitWidthImplType":{CONST:[1,3,1,""],PARAMETER:[1,3,1,""]},"brevitas.core.bit_width.ZeroLsbTruncBitWidth":{forward:[1,4,1,""]},"brevitas.core.function_wrapper":{CeilSte:[1,2,1,""],ClampMin:[1,2,1,""],ConstScalarClamp:[1,2,1,""],FloorSte:[1,2,1,""],Identity:[1,2,1,""],LogTwo:[1,2,1,""],OverBatchOverOutputChannelView:[1,2,1,""],OverBatchOverTensorView:[1,2,1,""],OverOutputChannelView:[1,2,1,""],OverTensorView:[1,2,1,""],PowerOfTwo:[1,2,1,""],RoundSte:[1,2,1,""],TensorClamp:[1,2,1,""],TensorClampSte:[1,2,1,""]},"brevitas.core.quant":{BinaryQuant:[1,2,1,""],PrescaledIntQuant:[1,2,1,""],PrescaledRestrictIntQuant:[1,2,1,""],QuantType:[1,2,1,""],RescalingIntQuant:[1,2,1,""],TernaryQuant:[1,2,1,""]},"brevitas.core.quant.QuantType":{BINARY:[1,3,1,""],FP:[1,3,1,""],INT:[1,3,1,""],TERNARY:[1,3,1,""]},"brevitas.core.quant.RescalingIntQuant":{scaling_init_from_min_max:[1,4,1,""]},"brevitas.core.restrict_val":{FloatToIntImplType:[1,2,1,""],RestrictValue:[1,2,1,""],RestrictValueOpImplType:[1,2,1,""],RestrictValueType:[1,2,1,""]},"brevitas.core.restrict_val.FloatToIntImplType":{CEIL:[1,3,1,""],FLOOR:[1,3,1,""],ROUND:[1,3,1,""]},"brevitas.core.restrict_val.RestrictValue":{restrict_value_op:[1,4,1,""]},"brevitas.core.restrict_val.RestrictValueOpImplType":{MATH:[1,3,1,""],TORCH_FN:[1,3,1,""],TORCH_MODULE:[1,3,1,""]},"brevitas.core.restrict_val.RestrictValueType":{FP:[1,3,1,""],INT:[1,3,1,""],LOG_FP:[1,3,1,""],POWER_OF_TWO:[1,3,1,""]},"brevitas.core.scaling":{AffineRescaling:[1,2,1,""],IntScaling:[1,2,1,""],ParameterStatsScaling:[1,2,1,""],PowerOfTwoIntScale:[1,2,1,""],RuntimeStatsScaling:[1,2,1,""],ScalingImplType:[1,2,1,""],SignedFpIntScale:[1,2,1,""],StandaloneScaling:[1,2,1,""],StatsScaling:[1,2,1,""],UnsignedFpIntScale:[1,2,1,""]},"brevitas.core.scaling.AffineRescaling":{forward:[1,4,1,""]},"brevitas.core.scaling.ScalingImplType":{AFFINE_STATS:[1,3,1,""],CONST:[1,3,1,""],HE:[1,3,1,""],OVERRIDE:[1,3,1,""],PARAMETER:[1,3,1,""],PARAMETER_FROM_STATS:[1,3,1,""],STATS:[1,3,1,""]},"brevitas.core.stats":{ParameterListStats:[1,2,1,""],StatsInputViewShapeImpl:[1,2,1,""],StatsOp:[1,2,1,""]},"brevitas.core.stats.StatsInputViewShapeImpl":{OVER_BATCH_OVER_OUTPUT_CHANNELS:[1,3,1,""],OVER_BATCH_OVER_TENSOR:[1,3,1,""],OVER_OUTPUT_CHANNELS:[1,3,1,""],OVER_TENSOR:[1,3,1,""]},"brevitas.core.stats.StatsOp":{AVE:[1,3,1,""],MAX:[1,3,1,""],MAX_AVE:[1,3,1,""],MEAN_LEARN_SIGMA_STD:[1,3,1,""],MEAN_SIGMA_STD:[1,3,1,""]},"brevitas.function":{autograd_ops:[2,0,0,"-"],ops:[2,0,0,"-"],ops_ste:[2,0,0,"-"],shape:[2,0,0,"-"]},"brevitas.function.autograd_ops":{binary_sign_ste_fn:[2,2,1,""],ceil_ste_fn:[2,2,1,""],floor_ste_fn:[2,2,1,""],round_ste_fn:[2,2,1,""],scalar_clamp_ste_fn:[2,2,1,""],tensor_clamp_ste_fn:[2,2,1,""],ternary_sign_ste_fn:[2,2,1,""]},"brevitas.function.autograd_ops.binary_sign_ste_fn":{backward:[2,4,1,""],forward:[2,4,1,""]},"brevitas.function.autograd_ops.ceil_ste_fn":{backward:[2,4,1,""],forward:[2,4,1,""]},"brevitas.function.autograd_ops.floor_ste_fn":{backward:[2,4,1,""],forward:[2,4,1,""]},"brevitas.function.autograd_ops.round_ste_fn":{backward:[2,4,1,""],forward:[2,4,1,""]},"brevitas.function.autograd_ops.scalar_clamp_ste_fn":{backward:[2,4,1,""],forward:[2,4,1,""]},"brevitas.function.autograd_ops.tensor_clamp_ste_fn":{backward:[2,4,1,""],forward:[2,4,1,""]},"brevitas.function.autograd_ops.ternary_sign_ste_fn":{backward:[2,4,1,""],forward:[2,4,1,""]},"brevitas.function.ops":{ceil_ste:[2,1,1,""],floor_ste:[2,1,1,""],tensor_clamp:[2,1,1,""]},"brevitas.loss":{base_loss:[3,0,0,"-"],weighted_bit_width:[3,0,0,"-"]},"brevitas.loss.base_loss":{SimpleBaseLoss:[3,2,1,""]},"brevitas.loss.base_loss.SimpleBaseLoss":{log:[3,4,1,""],register_hooks:[3,4,1,""],retrieve:[3,4,1,""],zero_accumulated_values:[3,4,1,""]},"brevitas.loss.weighted_bit_width":{ActivationBitWidthWeightedBySize:[3,2,1,""],BitWidthWeighted:[3,2,1,""],QuantLayerOutputBitWidthWeightedByOps:[3,2,1,""],WeightBitWidthWeightedBySize:[3,2,1,""]},"brevitas.loss.weighted_bit_width.ActivationBitWidthWeightedBySize":{register_hooks:[3,4,1,""]},"brevitas.loss.weighted_bit_width.BitWidthWeighted":{log:[3,4,1,""],register_hooks:[3,4,1,""],retrieve:[3,4,1,""],zero_accumulated_values:[3,4,1,""]},"brevitas.loss.weighted_bit_width.QuantLayerOutputBitWidthWeightedByOps":{register_hooks:[3,4,1,""]},"brevitas.loss.weighted_bit_width.WeightBitWidthWeightedBySize":{register_hooks:[3,4,1,""]},"brevitas.nn":{hadamard_classifier:[4,0,0,"-"],quant_accumulator:[4,0,0,"-"],quant_activation:[4,0,0,"-"],quant_avg_pool:[4,0,0,"-"],quant_bn:[4,0,0,"-"],quant_conv:[4,0,0,"-"],quant_layer:[4,0,0,"-"],quant_linear:[4,0,0,"-"],quant_scale_bias:[4,0,0,"-"]},"brevitas.nn.hadamard_classifier":{HadamardClassifier:[4,2,1,""]},"brevitas.nn.hadamard_classifier.HadamardClassifier":{forward:[4,4,1,""],max_output_bit_width:[4,4,1,""],state_dict:[4,4,1,""]},"brevitas.nn.quant_accumulator":{ClampQuantAccumulator:[4,2,1,""],QuantAccumulator:[4,2,1,""],TruncQuantAccumulator:[4,2,1,""]},"brevitas.nn.quant_accumulator.QuantAccumulator":{acc_quant_proxy:[4,4,1,""],forward:[4,4,1,""]},"brevitas.nn.quant_activation":{QuantActivation:[4,2,1,""],QuantHardTanh:[4,2,1,""],QuantReLU:[4,2,1,""],QuantSigmoid:[4,2,1,""],QuantTanh:[4,2,1,""]},"brevitas.nn.quant_activation.QuantActivation":{act_quant_proxy:[4,4,1,""],forward:[4,4,1,""],quant_act_scale:[4,4,1,""]},"brevitas.nn.quant_avg_pool":{QuantAvgPool2d:[4,2,1,""]},"brevitas.nn.quant_avg_pool.QuantAvgPool2d":{forward:[4,4,1,""],max_output_bit_width:[4,4,1,""]},"brevitas.nn.quant_bn":{BatchNorm2dToQuantScaleBias:[4,2,1,""],mul_add_from_bn:[4,1,1,""]},"brevitas.nn.quant_conv":{QuantConv2d:[4,2,1,""]},"brevitas.nn.quant_conv.QuantConv2d":{conv2d:[4,4,1,""],conv2d_same_padding:[4,4,1,""],forward:[4,4,1,""],int_weight:[4,4,1,""],max_output_bit_width:[4,4,1,""],merge_bn_in:[4,4,1,""],per_output_channel_broadcastable_shape:[4,4,1,""],quant_weight_scale:[4,4,1,""]},"brevitas.nn.quant_layer":{QuantLayer:[4,2,1,""]},"brevitas.nn.quant_layer.QuantLayer":{pack_output:[4,4,1,""],unpack_input:[4,4,1,""]},"brevitas.nn.quant_linear":{QuantLinear:[4,2,1,""]},"brevitas.nn.quant_linear.QuantLinear":{forward:[4,4,1,""],int_weight:[4,4,1,""],max_output_bit_width:[4,4,1,""],quant_weight_scale:[4,4,1,""]},"brevitas.nn.quant_scale_bias":{QuantScaleBias:[4,2,1,""],ScaleBias:[4,2,1,""]},"brevitas.nn.quant_scale_bias.QuantScaleBias":{forward:[4,4,1,""]},"brevitas.nn.quant_scale_bias.ScaleBias":{forward:[4,4,1,""]},"brevitas.proxy":{parameter_quant:[5,0,0,"-"],quant_proxy:[5,0,0,"-"],runtime_quant:[5,0,0,"-"]},"brevitas.proxy.parameter_quant":{BiasQuantProxy:[5,2,1,""],WeightQuantProxy:[5,2,1,""]},"brevitas.proxy.parameter_quant.BiasQuantProxy":{forward:[5,4,1,""]},"brevitas.proxy.parameter_quant.WeightQuantProxy":{add_tracked_parameter:[5,4,1,""],forward:[5,4,1,""],int_weight:[5,4,1,""],re_init_tensor_quant:[5,4,1,""]},"brevitas.proxy.quant_proxy":{QuantProxy:[5,2,1,""]},"brevitas.proxy.quant_proxy.QuantProxy":{state_dict:[5,4,1,""]},"brevitas.proxy.runtime_quant":{ActivationQuantProxy:[5,2,1,""],ClampQuantProxy:[5,2,1,""],FusedActivationQuantProxy:[5,2,1,""],TruncQuantProxy:[5,2,1,""]},"brevitas.proxy.runtime_quant.ActivationQuantProxy":{forward:[5,4,1,""]},"brevitas.proxy.runtime_quant.ClampQuantProxy":{forward:[5,4,1,""]},"brevitas.proxy.runtime_quant.TruncQuantProxy":{forward:[5,4,1,""]},"brevitas.quant_tensor":{QuantTensor:[0,2,1,""],pack_quant_tensor:[0,1,1,""]},"brevitas.quant_tensor.QuantTensor":{check_input_type:[0,4,1,""],check_scaling_factors_same:[0,4,1,""]},"brevitas.utils":{logging:[6,0,0,"-"],python_utils:[6,0,0,"-"],quant_utils:[6,0,0,"-"],torch_utils:[6,0,0,"-"]},"brevitas.utils.logging":{LogActivationBitWidth:[6,2,1,""],LogBitWidth:[6,2,1,""],LogWeightBitWidth:[6,2,1,""]},"brevitas.utils.logging.LogActivationBitWidth":{register_hooks:[6,4,1,""]},"brevitas.utils.logging.LogBitWidth":{register_hooks:[6,4,1,""]},"brevitas.utils.logging.LogWeightBitWidth":{register_hooks:[6,4,1,""]},"brevitas.utils.python_utils":{AutoName:[6,2,1,""]},"brevitas.utils.quant_utils":{has_learned_activation_bit_width:[6,1,1,""],has_learned_weight_bit_width:[6,1,1,""]},"brevitas.utils.torch_utils":{TupleSequential:[6,2,1,""]},"brevitas.utils.torch_utils.TupleSequential":{forward:[6,4,1,""],output:[6,4,1,""]},brevitas:{"function":[2,0,0,"-"],config:[0,0,0,"-"],core:[1,0,0,"-"],loss:[3,0,0,"-"],nn:[4,0,0,"-"],proxy:[5,0,0,"-"],quant_tensor:[0,0,0,"-"],utils:[6,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","attribute","Python attribute"],"4":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:attribute","4":"py:method"},terms:{"52587890625e":4,"abstract":[3,6],"boolean":2,"break":7,"case":[4,5,7],"class":[0,1,2,3,4,5,6,7],"const":[1,4,5],"default":[0,7],"enum":6,"float":[4,5,7],"function":[0,1,4,5,6,7,8],"import":7,"int":[1,4,5,7],"return":[1,2,4,5,7],"static":[0,1,2],"super":7,"true":[2,3,4,5,7],"while":[1,4,5,6,7],AVE:1,For:[4,5,7],The:[2,4,5,7],Useful:[4,5],With:7,__init__:7,abov:7,abs:7,acc_quant_proxi:4,accept:[2,4,5],accumul:7,accuraci:7,across:[4,5],act_quant_proxi:4,activ:7,activation_impl:5,activationbitwidthweightedbys:3,activationquantproxi:5,add:7,add_tracked_paramet:5,added:7,addit:7,addition:7,adher:7,adopt:[4,5],affin:[1,7],affine_onli:4,affine_shap:1,affine_stat:[1,4,5],affineresc:1,after:[2,5],afterward:[1,4,5,6,7],alessandro:7,alia:1,all:[1,2,4,5,6,7],allow:7,along:[5,7],alpha:7,also:2,altern:7,although:[1,4,5,6],ani:2,anyth:[4,5],api:7,appli:[2,5,7],approach:7,arg:6,argument:2,around:[4,5],as_averag:3,assembl:7,assum:[5,7],attribut:2,autograd:2,autograd_op:[0,8],automl:7,autonam:[1,6],avail:7,ave_learn_sigma_std:[4,5],ave_sigma_std:[4,5],averag:[4,5],avg:7,avgpool2d:4,avoid:7,awar:7,axes:7,back:[4,5],backprop:7,backpropag:2,backward:[2,7],base:[0,1,2,3,4,5,6],base_bit_width:7,base_loss:[0,8],batch:7,batchnorm2dtoquantscalebia:[4,7],becaus:7,befor:[5,7],behaviour:2,below:7,best:7,between:[2,4,5,7],bia:[4,5,7],bias:7,bias_bit_width:4,bias_narrow_rang:4,bias_quant_typ:4,biasquantproxi:5,binari:[1,7],binary_sign_ste_fn:2,binaryqu:1,bit:[4,5,7],bit_width:[0,4,5,7,8],bit_width_impl_overrid:[4,5],bit_width_impl_typ:[1,4,5],bit_width_init:1,bit_width_to_remov:1,bitwidthconst:[1,4,5],bitwidthimpltyp:[1,4,5],bitwidthparamet:[1,4,5],bitwidthweight:3,block:7,bn_bia:4,bn_ep:4,bn_mean:4,bn_var:4,bn_weight:4,bool:[4,5],both:[4,5,7],bound:[4,5],box:7,broadcast:5,buffer:[4,5],build:7,burden:7,call:[1,4,5,6,7],can:[2,4,5,7],care:[1,4,5,6],categor:7,ceil:1,ceil_st:2,ceil_ste_fn:2,ceilst:1,certain:7,chang:7,channel:[4,7],check_input_typ:0,check_scaling_factors_sam:0,clamp:[2,7],clamp_at_least_init_v:[1,4,5],clampmin:1,clampquantaccumul:4,clampquantproxi:5,classifi:7,clone:7,coeffici:7,collect:7,com:7,combin:7,compar:7,compat:7,complic:7,comput:[1,2,4,5,6,7],compute_output_bit_width:4,compute_output_scal:4,concat:[5,7],concaten:[5,7],config:8,conflict:7,consid:7,constant:7,constrain:7,constraint:[4,5,7],constscalarclamp:1,contain:[2,4,5,6,7],content:8,context:2,contribut:7,control:7,conv1:7,conv2:7,conv2d:[4,7],conv2d_same_pad:4,conv:4,converg:7,convolut:7,copi:7,core:[0,4,7,8],correctli:7,correspond:[2,4,5,7],cost:7,ctx:2,current:7,custom:7,data:7,dataflow:7,decai:7,def:7,defin:[1,2,4,5,6,7],denomin:7,depend:[4,7],depth:7,dequant:7,design:7,destin:[4,5],determin:[5,7],develop:7,dict:[4,5],dictionari:[4,5],did:2,differ:[4,5,7],differenti:[2,7],dilat:4,dimens:[4,5,7],directli:7,document:7,domain:7,done:7,download:7,drop:7,due:2,dure:2,each:[2,5,7],easili:7,effect:2,element:[2,7],els:[4,5],elsewher:[4,5],enumer:[1,6],env_to_bool:0,eps:4,equal:7,estim:7,estimat:2,evalu:7,everi:[1,2,4,5,6],exampl:[4,5,7],exist:7,explicit_resc:[4,5],expon:7,expos:7,extend:7,face:7,factor:[4,5,7],fals:[4,5,7],fc1:7,fc2:7,fc3:7,field:7,fine:7,first:[2,7],fixed_scal:4,float_to_int_impl:1,float_to_int_impl_typ:[1,4,5],floattointimpltyp:[1,4],floor:[1,2],floor_st:2,floor_ste_fn:2,floorst:1,flow:7,folder:7,follow:[2,7],former:[1,4,5,6],formula:2,forward:[1,2,4,5,6,7],full:7,function_wrapp:[0,4,8],fusedactivationquantproxi:5,futur:7,git:7,github:7,given:[2,4,5],goal:7,grad_i:2,gradient:[2,7],grain:7,graph:[4,5],group:4,gumbel:7,hadamard:7,hadamard_classifi:[0,8],hadamardclassifi:4,hardwar:7,has:[2,4,5],has_learned_activation_bit_width:6,has_learned_weight_bit_width:6,have:[2,7],here:7,hook:[1,4,5,6],how:7,howev:7,http:7,ident:1,idiomat:7,ignor:[1,4,5,6],implement:[2,4,5],implicit:7,impos:[4,5],in_channel:4,in_featur:4,includ:[4,5],index:7,infer:7,inform:7,initi:[4,5,7],input:[2,4,6,7],input_bit_width:[1,4,5],input_scal:5,instabl:[4,5],instanc:[1,4,5,6],instead:[1,4,5,6],int_scaling_impl:1,int_weight:[4,5],integ:7,interv:[4,5],intscal:1,involv:7,is_paramet:1,its:7,jit:[1,5,7],keep:7,keep_var:[4,5],kei:[4,5],kernel_s:4,kind:7,lab:7,lack:7,latest:7,latter:[1,4,5,6],layer:[4,5],layer_typ:3,learn:[4,5,7],lenet:7,level:7,leverag:7,librari:7,like:7,linear:4,link:7,list:[5,7],load:[4,5],log:[0,3,7,8],log_fp:[1,4],logactivationbitwidth:6,logbitwidth:6,logtwo:1,logweightbitwidth:6,look:7,loss:[0,7,8],low:7,lower:[4,5],ls_bit_width_to_trunc:[1,4,5],lsb_trunc_bit_width_impl_typ:[4,5],lsbtruncparameterbitwidth:1,made:7,magnitud:7,mainli:7,make:7,mani:2,math:1,max:[1,4,7],max_av:1,max_output_bit_width:4,max_overall_bit_width:[1,4,5],max_pool2d:7,max_val:[1,2,4,5,7],max_val_init:1,maximum:2,mean_learn_sigma_std:[1,4],mean_sigma_std:1,merge_bn_in:4,might:7,min:7,min_overall_bit_width:[1,4,5],min_val:[1,2,4,5],min_val_init:1,minimum:[2,4,5,7],mobile14:7,mobilenet:7,mod:6,model:[3,4,5,6],modul:[7,8],modular:7,more:7,mostli:7,ms_bit_width_to_clamp:[1,4,5],msb_clamp_bit_width_impl:1,msb_clamp_bit_width_impl_typ:[4,5],msbclampparameterbitwidth:1,much:7,mul_add_from_bn:4,multipl:7,must:2,name:[0,4,5,7],namedtupl:7,narrow_rang:[1,4,5],need:[1,2,4,5,6],needs_input_grad:2,neg:7,next:7,none:[4,5],norm:7,num_featur:4,number:[2,7],numer:[4,5],object:[1,3,4,6],offset:7,one:[1,4,5,6],onli:[4,5,7],oper:[2,4,5,7],ops:[0,8],ops_st:[0,8],ops_ste_n:[],ops_ste_o:[],option:[4,5,7],order:7,orthogon:7,other:[0,2,7],otherwis:[4,5],out:7,out_channel:4,out_featur:4,output:[2,4,6,7],output_bit_width:4,output_scal:4,over:[4,5,7],over_batch_over_output_channel:1,over_batch_over_tensor:1,over_output_channel:1,over_tensor:1,overal:7,overbatchoveroutputchannelview:1,overbatchovertensorview:1,overoutputchannelview:[1,4],overrid:[1,4,5],overridden:[1,2,4,5,6],override_pretrain:1,override_pretrained_bit_width:[4,5],overtensorview:1,pack_output:4,pack_quant_tensor:0,packag:[7,8],pad:4,padding_typ:4,paddingtyp:4,page:7,pappalardo:7,paramet:[1,2,4,5,7],parameter_from_stat:[1,4,5],parameter_qu:[0,8],parameter_shap:1,parameterliststat:1,parameterquantproxi:5,parameterstatssc:1,parametr:7,part:[4,5],partial:7,pass:[1,2,4,5,6,7],path:7,per:[4,7],per_channel_broadcastable_shap:[4,5],per_elem_op:4,per_output_channel_broadcastable_shap:4,perform:[1,2,4,5,6],persist:[4,5],pip:7,place:7,point:7,pool:[4,7],possibl:[4,5,7],power:7,power_of_two:1,poweroftwo:1,poweroftwointscal:1,practicion:7,pre:[4,5,7],preced:[4,5],precis:[4,5],prefix:[4,5],prescaledintqu:1,prescaledrestrictintqu:1,present:7,principl:7,probabl:7,progress:7,promot:7,propag:[4,5,7],properti:4,proxi:[0,7,8],proxylessna:7,purpos:7,put:7,python:2,python_util:[0,1,8],pytorch:[5,7],qnn:7,quant:[0,7,8],quant_accumul:[0,8],quant_act_scal:4,quant_activ:[0,8],quant_avg_pool:[0,8],quant_bn:[0,8],quant_conv:[0,3,8],quant_lay:[0,8],quant_linear:[0,3,8],quant_proxi:[0,8],quant_scale_bia:[0,8],quant_tensor:[4,7,8],quant_typ:[4,5,7],quant_util:[0,8],quant_weight_scal:4,quantaccumul:4,quantactiv:4,quantavgpool2d:[4,7],quantconv2d:[3,4,7],quanthardtanh:[4,7],quantiz:[4,5,7],quantlay:4,quantlayeroutputbitwidthweightedbyop:3,quantlenet:7,quantlinear:[3,4,7],quantproxi:5,quantrelu:[4,7],quantscalebia:[4,7],quantsigmoid:[4,7],quanttanh:[4,7],quanttensor:[0,7],quanttyp:[1,4,5,7],quit:7,rang:[4,5],re_init_tensor_qu:5,reach:[4,5],recip:[1,4,5,6],reduc:7,reg_coeff:3,regist:[1,4,5,6],register_hook:[3,6],regular:7,relax:[4,5,7],releas:7,relev:[4,5,7],relu1:7,relu2:7,relu3:7,relu4:7,relu6:7,remove_at_least_init_v:1,removebitwidthparamet:1,replac:[4,5,7],repo:7,repres:[2,7],represent:7,requir:[4,5],rescalingintqu:1,research:7,reshap:5,resort:7,respect:7,restrict:[4,5,7],restrict_bit_width_impl:1,restrict_bit_width_typ:[1,4,5],restrict_scaling_typ:[1,4,5],restrict_v:[0,8],restrict_value_op:1,restrict_value_op_impl_typ:1,restrict_value_typ:1,restrictvalu:1,restrictvalueopimpltyp:1,restrictvaluetyp:[1,4,5],result:7,retrain:7,retriev:[2,3],return_quant_tensor:4,rich:7,round:[1,4,7],round_st:7,round_ste_fn:2,roundst:1,routin:7,run:[1,4,5,6],runtim:1,runtime_qu:[0,8],runtimestatssc:1,same:[2,4,5,7],satur:7,save:2,scalar:4,scalar_clamp_ste_fn:2,scale:[0,4,5,8],scale_factor:7,scalebia:4,scaling_const:5,scaling_impl:1,scaling_impl_typ:[4,5],scaling_init:1,scaling_init_from_min_max:1,scaling_min_v:[1,4,5],scaling_overrid:[4,5],scaling_per_channel:[4,5],scaling_shap:5,scaling_stats_buffer_momentum:[4,5],scaling_stats_input_concat_dim:5,scaling_stats_input_view_shape_impl:[4,5],scaling_stats_op:[4,5],scaling_stats_permute_dim:[4,5],scaling_stats_reduce_dim:5,scaling_stats_sigma:[4,5],scalingimpltyp:[1,4,5],script:7,scriptmodul:[1,5],search:7,self:[4,7],sens:7,separ:7,sequenti:[6,7],set:[4,5,7],shape:[0,4,5,7,8],share:[4,5,7],should:[1,2,4,5,6],sigma:[1,4,5],sign:[1,4,5,7],signedfpintscal:1,silent:[1,4,5,6],simpl:7,simplebaseloss:3,sinc:[1,4,5,6],size:7,small:7,smaller:7,softmax:7,softwar:7,solut:7,some:[2,4,5,7],sort:7,space:7,specifi:[4,5],stage:7,standalon:7,standalonesc:1,standard:[4,7],stat:[0,4,5,8],state:[4,5],state_dict:[4,5],statist:[4,5,7],stats_buffer_init:1,stats_buffer_momentum:1,stats_input_concat_dim:1,stats_input_view_shape_impl:1,stats_op:1,stats_output_shap:1,stats_permute_dim:1,stats_reduce_dim:1,statsinputviewshapeimpl:1,statsop:[1,4,5],statsscal:1,ste:2,step:7,store:2,str:6,straight:[2,7],strategi:[4,5,7],strictli:7,stride:4,style:7,sub:7,subclass:[1,2,4,5,6],submodul:8,subpackag:8,subtract:7,symmetr:[4,5],take:[1,4,5,6],taken:[5,7],tensor:[0,1,2,5,7],tensor_clamp:2,tensor_clamp_impl:1,tensor_clamp_ste_fn:2,tensor_qu:5,tensorclamp:1,tensorclampst:1,ternari:[1,4,5,7],ternary_sign_ste_fn:2,ternary_threshold:5,ternaryqu:1,thei:7,them:[1,4,5,6,7],thi:[1,2,4,5,6,7],three:7,threshold:[1,4,5],through:[4,5,7],time:7,togeth:[5,7],top1:7,top5:7,torch:[1,2,4,5,6,7],torch_fn:1,torch_modul:1,torch_util:[0,8],toward:7,track:5,tracked_parameter_list:1,tracked_parameter_list_init:5,train:[4,5,7],tri:7,trough:2,trunc_at_least_init_v:[1,4,5],truncquantaccumul:4,truncquantproxi:5,tupl:[2,5,7],tuplesequenti:6,two:7,type:[1,2,4,5,7],unaffect:2,under:7,underli:7,unexpect:2,uniform:7,union:[4,5],unpack_input:4,unsignedfpintscal:1,upper:[4,5],upstream:7,use:7,used:[2,4,5,7],user:7,using:[2,7],usual:7,util:[0,1,8],valu:[2,4,5,7],variabl:7,variou:7,verbos:7,veri:7,verifi:7,version:7,view:7,wai:7,weight:[4,5,7],weight_bit_width:[4,7],weight_bit_width_impl_overrid:4,weight_bit_width_impl_typ:4,weight_max_overall_bit_width:4,weight_min_overall_bit_width:4,weight_narrow_rang:4,weight_override_pretrained_bit_width:4,weight_quant_overrid:4,weight_quant_typ:[4,7],weight_restrict_bit_width_typ:4,weight_restrict_scaling_typ:4,weight_scaling_const:4,weight_scaling_impl_typ:4,weight_scaling_min_v:4,weight_scaling_overrid:4,weight_scaling_per_output_channel:4,weight_scaling_stats_op:4,weight_scaling_stats_sigma:4,weight_ternary_threshold:4,weightbitwidthweightedbys:3,weighted_bit_width:[0,8],weightquantproxi:5,were:2,when:[2,4,5,7],whenev:5,where:7,whether:[2,4],which:[2,4,5,7],whole:[4,5],whose:7,width:[4,5,7],wise:7,within:[1,4,5,6],without:7,wrap:7,xilinx:7,you:7,zero_accumulated_valu:3,zero_hw_sentinel:1,zerolsbtruncbitwidth:1},titles:["brevitas package","brevitas.core package","brevitas.function package","brevitas.loss package","brevitas.nn package","brevitas.proxy package","brevitas.utils package","Brevitas","brevitas"],titleterms:{"function":2,audienc:7,author:7,autograd_op:2,base_loss:3,bit_width:1,brevita:[0,1,2,3,4,5,6,7,8],code:7,config:0,content:[0,1,2,3,4,5,6],core:1,dev:7,featur:7,from:7,function_wrapp:1,get:7,hadamard_classifi:4,implement:7,indic:7,instal:7,introduct:7,layer:7,log:6,loss:3,master:7,model:7,modul:[0,1,2,3,4,5,6],note:7,ops:2,ops_st:2,ops_ste_n:[],ops_ste_o:[],organ:7,packag:[0,1,2,3,4,5,6],parameter_qu:5,precis:7,pretrain:7,proxi:5,python_util:6,quant:1,quant_accumul:4,quant_activ:4,quant_avg_pool:4,quant_bn:4,quant_conv:4,quant_lay:4,quant_linear:4,quant_proxi:5,quant_scale_bia:4,quant_tensor:0,quant_util:6,requir:7,restrict_v:1,runtime_qu:5,scale:[1,7],shape:2,start:7,stat:1,submodul:[0,1,2,3,4,5,6],subpackag:0,support:7,tabl:7,target:7,torch_util:6,util:6,weighted_bit_width:3}})