??"
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	MLCMatMul
a"T
b"T

unique_key"T*num_args
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2"
num_argsint ("

input_rankint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*	2.4.0-rc02&tf_macos-v0.1-alpha2-AS-67-gf3595294ab8??
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2H* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:2H*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:H*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
'simple_rnn_10/simple_rnn_cell_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*8
shared_name)'simple_rnn_10/simple_rnn_cell_20/kernel
?
;simple_rnn_10/simple_rnn_cell_20/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_10/simple_rnn_cell_20/kernel*
_output_shapes
:	?*
dtype0
?
1simple_rnn_10/simple_rnn_cell_20/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*B
shared_name31simple_rnn_10/simple_rnn_cell_20/recurrent_kernel
?
Esimple_rnn_10/simple_rnn_cell_20/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_10/simple_rnn_cell_20/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
%simple_rnn_10/simple_rnn_cell_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%simple_rnn_10/simple_rnn_cell_20/bias
?
9simple_rnn_10/simple_rnn_cell_20/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_10/simple_rnn_cell_20/bias*
_output_shapes	
:?*
dtype0
?
'simple_rnn_11/simple_rnn_cell_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*8
shared_name)'simple_rnn_11/simple_rnn_cell_21/kernel
?
;simple_rnn_11/simple_rnn_cell_21/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_11/simple_rnn_cell_21/kernel*
_output_shapes
:	?2*
dtype0
?
1simple_rnn_11/simple_rnn_cell_21/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31simple_rnn_11/simple_rnn_cell_21/recurrent_kernel
?
Esimple_rnn_11/simple_rnn_cell_21/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_11/simple_rnn_cell_21/recurrent_kernel*
_output_shapes

:22*
dtype0
?
%simple_rnn_11/simple_rnn_cell_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%simple_rnn_11/simple_rnn_cell_21/bias
?
9simple_rnn_11/simple_rnn_cell_21/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_11/simple_rnn_cell_21/bias*
_output_shapes
:2*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2H*'
shared_nameAdam/dense_35/kernel/m
?
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

:2H*
dtype0
?
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:H*
dtype0
?
.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/m
?
BAdam/simple_rnn_10/simple_rnn_cell_20/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/m*
_output_shapes
:	?*
dtype0
?
8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*I
shared_name:8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/m
?
LAdam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
,Adam/simple_rnn_10/simple_rnn_cell_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/simple_rnn_10/simple_rnn_cell_20/bias/m
?
@Adam/simple_rnn_10/simple_rnn_cell_20/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_10/simple_rnn_cell_20/bias/m*
_output_shapes	
:?*
dtype0
?
.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*?
shared_name0.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/m
?
BAdam/simple_rnn_11/simple_rnn_cell_21/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/m*
_output_shapes
:	?2*
dtype0
?
8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/m
?
LAdam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/m*
_output_shapes

:22*
dtype0
?
,Adam/simple_rnn_11/simple_rnn_cell_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*=
shared_name.,Adam/simple_rnn_11/simple_rnn_cell_21/bias/m
?
@Adam/simple_rnn_11/simple_rnn_cell_21/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_11/simple_rnn_cell_21/bias/m*
_output_shapes
:2*
dtype0
?
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2H*'
shared_nameAdam/dense_35/kernel/v
?
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

:2H*
dtype0
?
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:H*
dtype0
?
.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/v
?
BAdam/simple_rnn_10/simple_rnn_cell_20/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/v*
_output_shapes
:	?*
dtype0
?
8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*I
shared_name:8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/v
?
LAdam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
,Adam/simple_rnn_10/simple_rnn_cell_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/simple_rnn_10/simple_rnn_cell_20/bias/v
?
@Adam/simple_rnn_10/simple_rnn_cell_20/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_10/simple_rnn_cell_20/bias/v*
_output_shapes	
:?*
dtype0
?
.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*?
shared_name0.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/v
?
BAdam/simple_rnn_11/simple_rnn_cell_21/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/v*
_output_shapes
:	?2*
dtype0
?
8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/v
?
LAdam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/v*
_output_shapes

:22*
dtype0
?
,Adam/simple_rnn_11/simple_rnn_cell_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*=
shared_name.,Adam/simple_rnn_11/simple_rnn_cell_21/bias/v
?
@Adam/simple_rnn_11/simple_rnn_cell_21/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_11/simple_rnn_cell_21/bias/v*
_output_shapes
:2*
dtype0

NoOpNoOp
?>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?>
value?>B?> B?>
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
?
0iter

1beta_1

2beta_2
	3decay
4learning_rate*m?+m?5m?6m?7m?8m?9m?:m?*v?+v?5v?6v?7v?8v?9v?:v?
 
8
50
61
72
83
94
:5
*6
+7
8
50
61
72
83
94
:5
*6
+7
?
	regularization_losses
;layer_regularization_losses
<layer_metrics

	variables
=non_trainable_variables
trainable_variables
>metrics

?layers
 
~

5kernel
6recurrent_kernel
7bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
 
 

50
61
72

50
61
72
?
regularization_losses
Dlayer_regularization_losses
Elayer_metrics
	variables
Fnon_trainable_variables
trainable_variables
Gmetrics

Hlayers

Istates
 
 
 
?
regularization_losses
Jlayer_regularization_losses
Klayer_metrics
	variables
Lnon_trainable_variables
trainable_variables
Mmetrics

Nlayers
 
 
 
?
regularization_losses
Olayer_regularization_losses
Player_metrics
	variables
Qnon_trainable_variables
trainable_variables
Rmetrics

Slayers
~

8kernel
9recurrent_kernel
:bias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
 
 

80
91
:2

80
91
:2
?
regularization_losses
Xlayer_regularization_losses
Ylayer_metrics
	variables
Znon_trainable_variables
 trainable_variables
[metrics

\layers

]states
 
 
 
?
"regularization_losses
^layer_regularization_losses
_layer_metrics
#	variables
`non_trainable_variables
$trainable_variables
ametrics

blayers
 
 
 
?
&regularization_losses
clayer_regularization_losses
dlayer_metrics
'	variables
enon_trainable_variables
(trainable_variables
fmetrics

glayers
[Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_35/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
?
,regularization_losses
hlayer_regularization_losses
ilayer_metrics
-	variables
jnon_trainable_variables
.trainable_variables
kmetrics

llayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'simple_rnn_10/simple_rnn_cell_20/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1simple_rnn_10/simple_rnn_cell_20/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_10/simple_rnn_cell_20/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'simple_rnn_11/simple_rnn_cell_21/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1simple_rnn_11/simple_rnn_cell_21/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_11/simple_rnn_cell_21/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

m0
n1
o2
1
0
1
2
3
4
5
6
 

50
61
72

50
61
72
?
@regularization_losses
player_regularization_losses
qlayer_metrics
A	variables
rnon_trainable_variables
Btrainable_variables
smetrics

tlayers
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 

80
91
:2

80
91
:2
?
Tregularization_losses
ulayer_regularization_losses
vlayer_metrics
U	variables
wnon_trainable_variables
Vtrainable_variables
xmetrics

ylayers
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ztotal
	{count
|	variables
}	keras_api
G
	~total
	count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

z0
{1

|	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

~0
1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/dense_35/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_10/simple_rnn_cell_20/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_11/simple_rnn_cell_21/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_10/simple_rnn_cell_20/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_11/simple_rnn_cell_21/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
#serving_default_simple_rnn_10_inputPlaceholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCall#serving_default_simple_rnn_10_input'simple_rnn_10/simple_rnn_cell_20/kernel%simple_rnn_10/simple_rnn_cell_20/bias1simple_rnn_10/simple_rnn_cell_20/recurrent_kernel'simple_rnn_11/simple_rnn_cell_21/kernel%simple_rnn_11/simple_rnn_cell_21/bias1simple_rnn_11/simple_rnn_cell_21/recurrent_kerneldense_35/kerneldense_35/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_242573
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp;simple_rnn_10/simple_rnn_cell_20/kernel/Read/ReadVariableOpEsimple_rnn_10/simple_rnn_cell_20/recurrent_kernel/Read/ReadVariableOp9simple_rnn_10/simple_rnn_cell_20/bias/Read/ReadVariableOp;simple_rnn_11/simple_rnn_cell_21/kernel/Read/ReadVariableOpEsimple_rnn_11/simple_rnn_cell_21/recurrent_kernel/Read/ReadVariableOp9simple_rnn_11/simple_rnn_cell_21/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOpBAdam/simple_rnn_10/simple_rnn_cell_20/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_10/simple_rnn_cell_20/bias/m/Read/ReadVariableOpBAdam/simple_rnn_11/simple_rnn_cell_21/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_11/simple_rnn_cell_21/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOpBAdam/simple_rnn_10/simple_rnn_cell_20/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_10/simple_rnn_cell_20/bias/v/Read/ReadVariableOpBAdam/simple_rnn_11/simple_rnn_cell_21/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_11/simple_rnn_cell_21/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_244409
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate'simple_rnn_10/simple_rnn_cell_20/kernel1simple_rnn_10/simple_rnn_cell_20/recurrent_kernel%simple_rnn_10/simple_rnn_cell_20/bias'simple_rnn_11/simple_rnn_cell_21/kernel1simple_rnn_11/simple_rnn_cell_21/recurrent_kernel%simple_rnn_11/simple_rnn_cell_21/biastotalcounttotal_1count_1total_2count_2Adam/dense_35/kernel/mAdam/dense_35/bias/m.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/m8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/m,Adam/simple_rnn_10/simple_rnn_cell_20/bias/m.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/m8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/m,Adam/simple_rnn_11/simple_rnn_cell_21/bias/mAdam/dense_35/kernel/vAdam/dense_35/bias/v.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/v8Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/v,Adam/simple_rnn_10/simple_rnn_cell_20/bias/v.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/v8Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/v,Adam/simple_rnn_11/simple_rnn_cell_21/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_244524??
?4
?
while_body_244015
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_21_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_21_matmul_readvariableop_resource<
8while_simple_rnn_cell_21_biasadd_readvariableop_resource=
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource??/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_21/MatMul/ReadVariableOp?0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_21_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_21/MatMul/ReadVariableOp?
while/simple_rnn_cell_21/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_21/MatMul?
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_21/BiasAddBiasAdd)while/simple_rnn_cell_21/MatMul:product:07while/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_21/BiasAdd?
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_21/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_21/MatMul_1?
while/simple_rnn_cell_21/addAddV2)while/simple_rnn_cell_21/BiasAdd:output:0+while/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/add?
while/simple_rnn_cell_21/TanhTanh while/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_21/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_21/Tanh:y:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_21_biasadd_readvariableop_resource:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_21_matmul_readvariableop_resource9while_simple_rnn_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_21/MatMul/ReadVariableOp.while/simple_rnn_cell_21/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
-sequential_30_simple_rnn_11_while_cond_240700T
Psequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_loop_counterZ
Vsequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_maximum_iterations1
-sequential_30_simple_rnn_11_while_placeholder3
/sequential_30_simple_rnn_11_while_placeholder_13
/sequential_30_simple_rnn_11_while_placeholder_2V
Rsequential_30_simple_rnn_11_while_less_sequential_30_simple_rnn_11_strided_slice_1l
hsequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_cond_240700___redundant_placeholder0l
hsequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_cond_240700___redundant_placeholder1l
hsequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_cond_240700___redundant_placeholder2l
hsequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_cond_240700___redundant_placeholder3.
*sequential_30_simple_rnn_11_while_identity
?
&sequential_30/simple_rnn_11/while/LessLess-sequential_30_simple_rnn_11_while_placeholderRsequential_30_simple_rnn_11_while_less_sequential_30_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: 2(
&sequential_30/simple_rnn_11/while/Less?
*sequential_30/simple_rnn_11/while/IdentityIdentity*sequential_30/simple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: 2,
*sequential_30/simple_rnn_11/while/Identity"a
*sequential_30_simple_rnn_11_while_identity3sequential_30/simple_rnn_11/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
G
+__inference_dropout_11_layer_call_fn_244132

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2423722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
while_cond_243242
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_243242___redundant_placeholder04
0while_while_cond_243242___redundant_placeholder14
0while_while_cond_243242___redundant_placeholder24
0while_while_cond_243242___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
b
F__inference_dropout_11_layer_call_and_return_conditional_losses_244122

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*'
_output_shapes
:?????????22	
dropoutd
IdentityIdentitydropout:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?H
?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_244081
inputs_05
1simple_rnn_cell_21_matmul_readvariableop_resource6
2simple_rnn_cell_21_biasadd_readvariableop_resource7
3simple_rnn_cell_21_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_21/BiasAdd/ReadVariableOp?(simple_rnn_cell_21/MatMul/ReadVariableOp?*simple_rnn_cell_21/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_21_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_21/MatMul/ReadVariableOp?
simple_rnn_cell_21/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul?
)simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_21/BiasAdd/ReadVariableOp?
simple_rnn_cell_21/BiasAddBiasAdd#simple_rnn_cell_21/MatMul:product:01simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/BiasAdd?
*simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_21/MatMul_1/ReadVariableOp?
simple_rnn_cell_21/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul_1?
simple_rnn_cell_21/addAddV2#simple_rnn_cell_21/BiasAdd:output:0%simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/add?
simple_rnn_cell_21/TanhTanhsimple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_21_matmul_readvariableop_resource2simple_rnn_cell_21_biasadd_readvariableop_resource3simple_rnn_cell_21_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_244015*
condR
while_cond_244014*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_21/BiasAdd/ReadVariableOp)^simple_rnn_cell_21/MatMul/ReadVariableOp+^simple_rnn_cell_21/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2V
)simple_rnn_cell_21/BiasAdd/ReadVariableOp)simple_rnn_cell_21/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_21/MatMul/ReadVariableOp(simple_rnn_cell_21/MatMul/ReadVariableOp2X
*simple_rnn_cell_21/MatMul_1/ReadVariableOp*simple_rnn_cell_21/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_244191

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpw
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp
MatMul_1	MLCMatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
I__inference_sequential_30_layer_call_and_return_conditional_losses_242475

inputs
simple_rnn_10_242451
simple_rnn_10_242453
simple_rnn_10_242455
simple_rnn_11_242460
simple_rnn_11_242462
simple_rnn_11_242464
dense_35_242469
dense_35_242471
identity?? dense_35/StatefulPartitionedCall?%simple_rnn_10/StatefulPartitionedCall?%simple_rnn_11/StatefulPartitionedCall?
%simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_10_242451simple_rnn_10_242453simple_rnn_10_242455*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_2419152'
%simple_rnn_10/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall.simple_rnn_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_2420622
activation_25/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall&activation_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2420792
dropout_10/PartitionedCall?
%simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0simple_rnn_11_242460simple_rnn_11_242462simple_rnn_11_242464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_2422082'
%simple_rnn_11/StatefulPartitionedCall?
activation_26/PartitionedCallPartitionedCall.simple_rnn_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_26_layer_call_and_return_conditional_losses_2423552
activation_26/PartitionedCall?
dropout_11/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2423722
dropout_11/PartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_35_242469dense_35_242471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_2424012"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_35/StatefulPartitionedCall&^simple_rnn_10/StatefulPartitionedCall&^simple_rnn_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%simple_rnn_10/StatefulPartitionedCall%simple_rnn_10/StatefulPartitionedCall2N
%simple_rnn_11/StatefulPartitionedCall%simple_rnn_11/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?H
?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_242208

inputs5
1simple_rnn_cell_21_matmul_readvariableop_resource6
2simple_rnn_cell_21_biasadd_readvariableop_resource7
3simple_rnn_cell_21_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_21/BiasAdd/ReadVariableOp?(simple_rnn_cell_21/MatMul/ReadVariableOp?*simple_rnn_cell_21/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:???????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_21_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_21/MatMul/ReadVariableOp?
simple_rnn_cell_21/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul?
)simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_21/BiasAdd/ReadVariableOp?
simple_rnn_cell_21/BiasAddBiasAdd#simple_rnn_cell_21/MatMul:product:01simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/BiasAdd?
*simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_21/MatMul_1/ReadVariableOp?
simple_rnn_cell_21/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul_1?
simple_rnn_cell_21/addAddV2#simple_rnn_cell_21/BiasAdd:output:0%simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/add?
simple_rnn_cell_21/TanhTanhsimple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_21_matmul_readvariableop_resource2simple_rnn_cell_21_biasadd_readvariableop_resource3simple_rnn_cell_21_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_242142*
condR
while_cond_242141*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_21/BiasAdd/ReadVariableOp)^simple_rnn_cell_21/MatMul/ReadVariableOp+^simple_rnn_cell_21/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::2V
)simple_rnn_cell_21/BiasAdd/ReadVariableOp)simple_rnn_cell_21/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_21/MatMul/ReadVariableOp(simple_rnn_cell_21/MatMul/ReadVariableOp2X
*simple_rnn_cell_21/MatMul_1/ReadVariableOp*simple_rnn_cell_21/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
simple_rnn_11_while_cond_2427328
4simple_rnn_11_while_simple_rnn_11_while_loop_counter>
:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations#
simple_rnn_11_while_placeholder%
!simple_rnn_11_while_placeholder_1%
!simple_rnn_11_while_placeholder_2:
6simple_rnn_11_while_less_simple_rnn_11_strided_slice_1P
Lsimple_rnn_11_while_simple_rnn_11_while_cond_242732___redundant_placeholder0P
Lsimple_rnn_11_while_simple_rnn_11_while_cond_242732___redundant_placeholder1P
Lsimple_rnn_11_while_simple_rnn_11_while_cond_242732___redundant_placeholder2P
Lsimple_rnn_11_while_simple_rnn_11_while_cond_242732___redundant_placeholder3 
simple_rnn_11_while_identity
?
simple_rnn_11/while/LessLesssimple_rnn_11_while_placeholder6simple_rnn_11_while_less_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_11/while/Less?
simple_rnn_11/while/IdentityIdentitysimple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_11/while/Identity"E
simple_rnn_11_while_identity%simple_rnn_11/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?H
?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243555

inputs5
1simple_rnn_cell_20_matmul_readvariableop_resource6
2simple_rnn_cell_20_biasadd_readvariableop_resource7
3simple_rnn_cell_20_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_20/BiasAdd/ReadVariableOp?(simple_rnn_cell_20/MatMul/ReadVariableOp?*simple_rnn_cell_20/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_20/MatMul/ReadVariableOp?
simple_rnn_cell_20/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul?
)simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_20/BiasAdd/ReadVariableOp?
simple_rnn_cell_20/BiasAddBiasAdd#simple_rnn_cell_20/MatMul:product:01simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/BiasAdd?
*simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_20/MatMul_1/ReadVariableOp?
simple_rnn_cell_20/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul_1?
simple_rnn_cell_20/addAddV2#simple_rnn_cell_20/BiasAdd:output:0%simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/add?
simple_rnn_cell_20/TanhTanhsimple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_20_matmul_readvariableop_resource2simple_rnn_cell_20_biasadd_readvariableop_resource3simple_rnn_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_243489*
condR
while_cond_243488*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_20/BiasAdd/ReadVariableOp)^simple_rnn_cell_20/MatMul/ReadVariableOp+^simple_rnn_cell_20/MatMul_1/ReadVariableOp^while*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::2V
)simple_rnn_cell_20/BiasAdd/ReadVariableOp)simple_rnn_cell_20/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_20/MatMul/ReadVariableOp(simple_rnn_cell_20/MatMul/ReadVariableOp2X
*simple_rnn_cell_20/MatMul_1/ReadVariableOp*simple_rnn_cell_20/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_241354

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOpv
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1	MLCMatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????22
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?
?
while_cond_241848
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241848___redundant_placeholder04
0while_while_cond_241848___redundant_placeholder14
0while_while_cond_241848___redundant_placeholder24
0while_while_cond_241848___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?T
?
-sequential_30_simple_rnn_11_while_body_240701T
Psequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_loop_counterZ
Vsequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_maximum_iterations1
-sequential_30_simple_rnn_11_while_placeholder3
/sequential_30_simple_rnn_11_while_placeholder_13
/sequential_30_simple_rnn_11_while_placeholder_2S
Osequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_strided_slice_1_0?
?sequential_30_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0Y
Usequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resource_0Z
Vsequential_30_simple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resource_0[
Wsequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0.
*sequential_30_simple_rnn_11_while_identity0
,sequential_30_simple_rnn_11_while_identity_10
,sequential_30_simple_rnn_11_while_identity_20
,sequential_30_simple_rnn_11_while_identity_30
,sequential_30_simple_rnn_11_while_identity_4Q
Msequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_strided_slice_1?
?sequential_30_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorW
Ssequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resourceX
Tsequential_30_simple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resourceY
Usequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource??Ksequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?Jsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?Lsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
Ssequential_30/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2U
Ssequential_30/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Esequential_30/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_30_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0-sequential_30_simple_rnn_11_while_placeholder\sequential_30/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02G
Esequential_30/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem?
Jsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOpUsequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02L
Jsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?
;sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul	MLCMatMulLsequential_30/simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22=
;sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul?
Ksequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOpVsequential_30_simple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02M
Ksequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
<sequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAddBiasAddEsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul:product:0Ssequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22>
<sequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd?
Lsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOpWsequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02N
Lsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
=sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1	MLCMatMul/sequential_30_simple_rnn_11_while_placeholder_2Tsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22?
=sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1?
8sequential_30/simple_rnn_11/while/simple_rnn_cell_21/addAddV2Esequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd:output:0Gsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22:
8sequential_30/simple_rnn_11/while/simple_rnn_cell_21/add?
9sequential_30/simple_rnn_11/while/simple_rnn_cell_21/TanhTanh<sequential_30/simple_rnn_11/while/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22;
9sequential_30/simple_rnn_11/while/simple_rnn_cell_21/Tanh?
Fsequential_30/simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_30_simple_rnn_11_while_placeholder_1-sequential_30_simple_rnn_11_while_placeholder=sequential_30/simple_rnn_11/while/simple_rnn_cell_21/Tanh:y:0*
_output_shapes
: *
element_dtype02H
Fsequential_30/simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem?
'sequential_30/simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_30/simple_rnn_11/while/add/y?
%sequential_30/simple_rnn_11/while/addAddV2-sequential_30_simple_rnn_11_while_placeholder0sequential_30/simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: 2'
%sequential_30/simple_rnn_11/while/add?
)sequential_30/simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_30/simple_rnn_11/while/add_1/y?
'sequential_30/simple_rnn_11/while/add_1AddV2Psequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_loop_counter2sequential_30/simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2)
'sequential_30/simple_rnn_11/while/add_1?
*sequential_30/simple_rnn_11/while/IdentityIdentity+sequential_30/simple_rnn_11/while/add_1:z:0L^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOpM^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_30/simple_rnn_11/while/Identity?
,sequential_30/simple_rnn_11/while/Identity_1IdentityVsequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_while_maximum_iterationsL^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOpM^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2.
,sequential_30/simple_rnn_11/while/Identity_1?
,sequential_30/simple_rnn_11/while/Identity_2Identity)sequential_30/simple_rnn_11/while/add:z:0L^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOpM^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2.
,sequential_30/simple_rnn_11/while/Identity_2?
,sequential_30/simple_rnn_11/while/Identity_3IdentityVsequential_30/simple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0L^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOpM^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2.
,sequential_30/simple_rnn_11/while/Identity_3?
,sequential_30/simple_rnn_11/while/Identity_4Identity=sequential_30/simple_rnn_11/while/simple_rnn_cell_21/Tanh:y:0L^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOpM^sequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22.
,sequential_30/simple_rnn_11/while/Identity_4"a
*sequential_30_simple_rnn_11_while_identity3sequential_30/simple_rnn_11/while/Identity:output:0"e
,sequential_30_simple_rnn_11_while_identity_15sequential_30/simple_rnn_11/while/Identity_1:output:0"e
,sequential_30_simple_rnn_11_while_identity_25sequential_30/simple_rnn_11/while/Identity_2:output:0"e
,sequential_30_simple_rnn_11_while_identity_35sequential_30/simple_rnn_11/while/Identity_3:output:0"e
,sequential_30_simple_rnn_11_while_identity_45sequential_30/simple_rnn_11/while/Identity_4:output:0"?
Msequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_strided_slice_1Osequential_30_simple_rnn_11_while_sequential_30_simple_rnn_11_strided_slice_1_0"?
Tsequential_30_simple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resourceVsequential_30_simple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resource_0"?
Usequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resourceWsequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0"?
Ssequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resourceUsequential_30_simple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resource_0"?
?sequential_30_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor?sequential_30_simple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2?
Ksequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpKsequential_30/simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp2?
Jsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOpJsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp2?
Lsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOpLsequential_30/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_241727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241727___redundant_placeholder04
0while_while_cond_241727___redundant_placeholder14
0while_while_cond_241727___redundant_placeholder24
0while_while_cond_241727___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
e
I__inference_activation_26_layer_call_and_return_conditional_losses_242355

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?D
?
simple_rnn_11_while_body_2429688
4simple_rnn_11_while_simple_rnn_11_while_loop_counter>
:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations#
simple_rnn_11_while_placeholder%
!simple_rnn_11_while_placeholder_1%
!simple_rnn_11_while_placeholder_27
3simple_rnn_11_while_simple_rnn_11_strided_slice_1_0s
osimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0K
Gsimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resource_0L
Hsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resource_0M
Isimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0 
simple_rnn_11_while_identity"
simple_rnn_11_while_identity_1"
simple_rnn_11_while_identity_2"
simple_rnn_11_while_identity_3"
simple_rnn_11_while_identity_45
1simple_rnn_11_while_simple_rnn_11_strided_slice_1q
msimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorI
Esimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resourceJ
Fsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resourceK
Gsimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource??=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
Esimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2G
Esimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape?
7simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_11_while_placeholderNsimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype029
7simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem?
<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02>
<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?
-simple_rnn_11/while/simple_rnn_cell_21/MatMul	MLCMatMul>simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22/
-simple_rnn_11/while/simple_rnn_cell_21/MatMul?
=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02?
=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
.simple_rnn_11/while/simple_rnn_cell_21/BiasAddBiasAdd7simple_rnn_11/while/simple_rnn_cell_21/MatMul:product:0Esimple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.simple_rnn_11/while/simple_rnn_cell_21/BiasAdd?
>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02@
>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1	MLCMatMul!simple_rnn_11_while_placeholder_2Fsimple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????221
/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1?
*simple_rnn_11/while/simple_rnn_cell_21/addAddV27simple_rnn_11/while/simple_rnn_cell_21/BiasAdd:output:09simple_rnn_11/while/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22,
*simple_rnn_11/while/simple_rnn_cell_21/add?
+simple_rnn_11/while/simple_rnn_cell_21/TanhTanh.simple_rnn_11/while/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22-
+simple_rnn_11/while/simple_rnn_cell_21/Tanh?
8simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_11_while_placeholder_1simple_rnn_11_while_placeholder/simple_rnn_11/while/simple_rnn_cell_21/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemx
simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_11/while/add/y?
simple_rnn_11/while/addAddV2simple_rnn_11_while_placeholder"simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_11/while/add|
simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_11/while/add_1/y?
simple_rnn_11/while/add_1AddV24simple_rnn_11_while_simple_rnn_11_while_loop_counter$simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_11/while/add_1?
simple_rnn_11/while/IdentityIdentitysimple_rnn_11/while/add_1:z:0>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_11/while/Identity?
simple_rnn_11/while/Identity_1Identity:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_11/while/Identity_1?
simple_rnn_11/while/Identity_2Identitysimple_rnn_11/while/add:z:0>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_11/while/Identity_2?
simple_rnn_11/while/Identity_3IdentityHsimple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_11/while/Identity_3?
simple_rnn_11/while/Identity_4Identity/simple_rnn_11/while/simple_rnn_cell_21/Tanh:y:0>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22 
simple_rnn_11/while/Identity_4"E
simple_rnn_11_while_identity%simple_rnn_11/while/Identity:output:0"I
simple_rnn_11_while_identity_1'simple_rnn_11/while/Identity_1:output:0"I
simple_rnn_11_while_identity_2'simple_rnn_11/while/Identity_2:output:0"I
simple_rnn_11_while_identity_3'simple_rnn_11/while/Identity_3:output:0"I
simple_rnn_11_while_identity_4'simple_rnn_11/while/Identity_4:output:0"h
1simple_rnn_11_while_simple_rnn_11_strided_slice_13simple_rnn_11_while_simple_rnn_11_strided_slice_1_0"?
Fsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resourceHsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resource_0"?
Gsimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resourceIsimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0"?
Esimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resourceGsimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resource_0"?
msimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorosimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2~
=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp2|
<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp2?
>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?4
?
while_body_243243
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_20_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_20_matmul_readvariableop_resource<
8while_simple_rnn_cell_20_biasadd_readvariableop_resource=
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource??/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_20/MatMul/ReadVariableOp?0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_20/MatMul/ReadVariableOp?
while/simple_rnn_cell_20/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_20/MatMul?
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_20/BiasAddBiasAdd)while/simple_rnn_cell_20/MatMul:product:07while/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_20/BiasAdd?
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_20/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_20/MatMul_1?
while/simple_rnn_cell_20/addAddV2)while/simple_rnn_cell_20/BiasAdd:output:0+while/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/add?
while/simple_rnn_cell_20/TanhTanh while/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_20/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_20/Tanh:y:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_20_biasadd_readvariableop_resource:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_20_matmul_readvariableop_resource9while_simple_rnn_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_20/MatMul/ReadVariableOp.while/simple_rnn_cell_20/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_243376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_243376___redundant_placeholder04
0while_while_cond_243376___redundant_placeholder14
0while_while_cond_243376___redundant_placeholder24
0while_while_cond_243376___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
D__inference_dense_35_layer_call_and_return_conditional_losses_244148

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:2H*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_244127

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?H
?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243309
inputs_05
1simple_rnn_cell_20_matmul_readvariableop_resource6
2simple_rnn_cell_20_biasadd_readvariableop_resource7
3simple_rnn_cell_20_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_20/BiasAdd/ReadVariableOp?(simple_rnn_cell_20/MatMul/ReadVariableOp?*simple_rnn_cell_20/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_20/MatMul/ReadVariableOp?
simple_rnn_cell_20/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul?
)simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_20/BiasAdd/ReadVariableOp?
simple_rnn_cell_20/BiasAddBiasAdd#simple_rnn_cell_20/MatMul:product:01simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/BiasAdd?
*simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_20/MatMul_1/ReadVariableOp?
simple_rnn_cell_20/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul_1?
simple_rnn_cell_20/addAddV2#simple_rnn_cell_20/BiasAdd:output:0%simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/add?
simple_rnn_cell_20/TanhTanhsimple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_20_matmul_readvariableop_resource2simple_rnn_cell_20_biasadd_readvariableop_resource3simple_rnn_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_243243*
condR
while_cond_243242*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_20/BiasAdd/ReadVariableOp)^simple_rnn_cell_20/MatMul/ReadVariableOp+^simple_rnn_cell_20/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_20/BiasAdd/ReadVariableOp)simple_rnn_cell_20/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_20/MatMul/ReadVariableOp(simple_rnn_cell_20/MatMul/ReadVariableOp2X
*simple_rnn_cell_20/MatMul_1/ReadVariableOp*simple_rnn_cell_20/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
.__inference_simple_rnn_11_layer_call_fn_243857

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_2423202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?4
?
while_body_242142
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_21_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_21_matmul_readvariableop_resource<
8while_simple_rnn_cell_21_biasadd_readvariableop_resource=
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource??/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_21/MatMul/ReadVariableOp?0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_21_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_21/MatMul/ReadVariableOp?
while/simple_rnn_cell_21/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_21/MatMul?
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_21/BiasAddBiasAdd)while/simple_rnn_cell_21/MatMul:product:07while/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_21/BiasAdd?
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_21/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_21/MatMul_1?
while/simple_rnn_cell_21/addAddV2)while/simple_rnn_cell_21/BiasAdd:output:0+while/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/add?
while/simple_rnn_cell_21/TanhTanh while/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_21/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_21/Tanh:y:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_21_biasadd_readvariableop_resource:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_21_matmul_readvariableop_resource9while_simple_rnn_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_21/MatMul/ReadVariableOp.while/simple_rnn_cell_21/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_sequential_30_layer_call_and_return_conditional_losses_242418
simple_rnn_10_input
simple_rnn_10_242050
simple_rnn_10_242052
simple_rnn_10_242054
simple_rnn_11_242343
simple_rnn_11_242345
simple_rnn_11_242347
dense_35_242412
dense_35_242414
identity?? dense_35/StatefulPartitionedCall?%simple_rnn_10/StatefulPartitionedCall?%simple_rnn_11/StatefulPartitionedCall?
%simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_10_inputsimple_rnn_10_242050simple_rnn_10_242052simple_rnn_10_242054*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_2419152'
%simple_rnn_10/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall.simple_rnn_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_2420622
activation_25/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall&activation_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2420792
dropout_10/PartitionedCall?
%simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0simple_rnn_11_242343simple_rnn_11_242345simple_rnn_11_242347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_2422082'
%simple_rnn_11/StatefulPartitionedCall?
activation_26/PartitionedCallPartitionedCall.simple_rnn_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_26_layer_call_and_return_conditional_losses_2423552
activation_26/PartitionedCall?
dropout_11/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2423722
dropout_11/PartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_35_242412dense_35_242414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_2424012"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_35/StatefulPartitionedCall&^simple_rnn_10/StatefulPartitionedCall&^simple_rnn_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%simple_rnn_10/StatefulPartitionedCall%simple_rnn_10/StatefulPartitionedCall2N
%simple_rnn_11/StatefulPartitionedCall%simple_rnn_11/StatefulPartitionedCall:a ]
,
_output_shapes
:??????????
-
_user_specified_namesimple_rnn_10_input
?
~
)__inference_dense_35_layer_call_fn_244157

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_2424012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_244253

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOpv
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1	MLCMatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????22
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?
?
.__inference_simple_rnn_10_layer_call_fn_243331
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_2412792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?4
?
while_body_243131
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_20_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_20_matmul_readvariableop_resource<
8while_simple_rnn_cell_20_biasadd_readvariableop_resource=
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource??/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_20/MatMul/ReadVariableOp?0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_20/MatMul/ReadVariableOp?
while/simple_rnn_cell_20/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_20/MatMul?
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_20/BiasAddBiasAdd)while/simple_rnn_cell_20/MatMul:product:07while/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_20/BiasAdd?
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_20/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_20/MatMul_1?
while/simple_rnn_cell_20/addAddV2)while/simple_rnn_cell_20/BiasAdd:output:0+while/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/add?
while/simple_rnn_cell_20/TanhTanh while/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_20/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_20/Tanh:y:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_20_biasadd_readvariableop_resource:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_20_matmul_readvariableop_resource9while_simple_rnn_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_20/MatMul/ReadVariableOp.while/simple_rnn_cell_20/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?#
?
while_body_241611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_21_241633_0%
!while_simple_rnn_cell_21_241635_0%
!while_simple_rnn_cell_21_241637_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_21_241633#
while_simple_rnn_cell_21_241635#
while_simple_rnn_cell_21_241637??0while/simple_rnn_cell_21/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
0while/simple_rnn_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_21_241633_0!while_simple_rnn_cell_21_241635_0!while_simple_rnn_cell_21_241637_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_24133722
0while/simple_rnn_cell_21/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_21/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/simple_rnn_cell_21/StatefulPartitionedCall:output:11^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_21_241633!while_simple_rnn_cell_21_241633_0"D
while_simple_rnn_cell_21_241635!while_simple_rnn_cell_21_241635_0"D
while_simple_rnn_cell_21_241637!while_simple_rnn_cell_21_241637_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2d
0while/simple_rnn_cell_21/StatefulPartitionedCall0while/simple_rnn_cell_21/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
??
?
I__inference_sequential_30_layer_call_and_return_conditional_losses_243043

inputsC
?simple_rnn_10_simple_rnn_cell_20_matmul_readvariableop_resourceD
@simple_rnn_10_simple_rnn_cell_20_biasadd_readvariableop_resourceE
Asimple_rnn_10_simple_rnn_cell_20_matmul_1_readvariableop_resourceC
?simple_rnn_11_simple_rnn_cell_21_matmul_readvariableop_resourceD
@simple_rnn_11_simple_rnn_cell_21_biasadd_readvariableop_resourceE
Asimple_rnn_11_simple_rnn_cell_21_matmul_1_readvariableop_resource.
*dense_35_mlcmatmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??dense_35/BiasAdd/ReadVariableOp?!dense_35/MLCMatMul/ReadVariableOp?7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp?6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp?8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp?simple_rnn_10/while?7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp?6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp?8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp?simple_rnn_11/while`
simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_10/Shape?
!simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!simple_rnn_10/strided_slice/stack?
#simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_10/strided_slice/stack_1?
#simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_10/strided_slice/stack_2?
simple_rnn_10/strided_sliceStridedSlicesimple_rnn_10/Shape:output:0*simple_rnn_10/strided_slice/stack:output:0,simple_rnn_10/strided_slice/stack_1:output:0,simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_10/strided_slicey
simple_rnn_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_10/zeros/mul/y?
simple_rnn_10/zeros/mulMul$simple_rnn_10/strided_slice:output:0"simple_rnn_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_10/zeros/mul{
simple_rnn_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_10/zeros/Less/y?
simple_rnn_10/zeros/LessLesssimple_rnn_10/zeros/mul:z:0#simple_rnn_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_10/zeros/Less
simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_10/zeros/packed/1?
simple_rnn_10/zeros/packedPack$simple_rnn_10/strided_slice:output:0%simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_10/zeros/packed{
simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_10/zeros/Const?
simple_rnn_10/zerosFill#simple_rnn_10/zeros/packed:output:0"simple_rnn_10/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_10/zeros?
simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_10/transpose/perm?
simple_rnn_10/transpose	Transposeinputs%simple_rnn_10/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
simple_rnn_10/transposey
simple_rnn_10/Shape_1Shapesimple_rnn_10/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_10/Shape_1?
#simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_10/strided_slice_1/stack?
%simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_1/stack_1?
%simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_1/stack_2?
simple_rnn_10/strided_slice_1StridedSlicesimple_rnn_10/Shape_1:output:0,simple_rnn_10/strided_slice_1/stack:output:0.simple_rnn_10/strided_slice_1/stack_1:output:0.simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_10/strided_slice_1?
)simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)simple_rnn_10/TensorArrayV2/element_shape?
simple_rnn_10/TensorArrayV2TensorListReserve2simple_rnn_10/TensorArrayV2/element_shape:output:0&simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_10/TensorArrayV2?
Csimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2E
Csimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape?
5simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_10/transpose:y:0Lsimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5simple_rnn_10/TensorArrayUnstack/TensorListFromTensor?
#simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_10/strided_slice_2/stack?
%simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_2/stack_1?
%simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_2/stack_2?
simple_rnn_10/strided_slice_2StridedSlicesimple_rnn_10/transpose:y:0,simple_rnn_10/strided_slice_2/stack:output:0.simple_rnn_10/strided_slice_2/stack_1:output:0.simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_10/strided_slice_2?
6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp?simple_rnn_10_simple_rnn_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype028
6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp?
'simple_rnn_10/simple_rnn_cell_20/MatMul	MLCMatMul&simple_rnn_10/strided_slice_2:output:0>simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_10/simple_rnn_cell_20/MatMul?
7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_10_simple_rnn_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
(simple_rnn_10/simple_rnn_cell_20/BiasAddBiasAdd1simple_rnn_10/simple_rnn_cell_20/MatMul:product:0?simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_10/simple_rnn_cell_20/BiasAdd?
8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_10_simple_rnn_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
)simple_rnn_10/simple_rnn_cell_20/MatMul_1	MLCMatMulsimple_rnn_10/zeros:output:0@simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_10/simple_rnn_cell_20/MatMul_1?
$simple_rnn_10/simple_rnn_cell_20/addAddV21simple_rnn_10/simple_rnn_cell_20/BiasAdd:output:03simple_rnn_10/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2&
$simple_rnn_10/simple_rnn_cell_20/add?
%simple_rnn_10/simple_rnn_cell_20/TanhTanh(simple_rnn_10/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2'
%simple_rnn_10/simple_rnn_cell_20/Tanh?
+simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2-
+simple_rnn_10/TensorArrayV2_1/element_shape?
simple_rnn_10/TensorArrayV2_1TensorListReserve4simple_rnn_10/TensorArrayV2_1/element_shape:output:0&simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_10/TensorArrayV2_1j
simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_10/time?
&simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&simple_rnn_10/while/maximum_iterations?
 simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 simple_rnn_10/while/loop_counter?
simple_rnn_10/whileWhile)simple_rnn_10/while/loop_counter:output:0/simple_rnn_10/while/maximum_iterations:output:0simple_rnn_10/time:output:0&simple_rnn_10/TensorArrayV2_1:handle:0simple_rnn_10/zeros:output:0&simple_rnn_10/strided_slice_1:output:0Esimple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_10_simple_rnn_cell_20_matmul_readvariableop_resource@simple_rnn_10_simple_rnn_cell_20_biasadd_readvariableop_resourceAsimple_rnn_10_simple_rnn_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*+
body#R!
simple_rnn_10_while_body_242858*+
cond#R!
simple_rnn_10_while_cond_242857*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_10/while?
>simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape?
0simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_10/while:output:3Gsimple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype022
0simple_rnn_10/TensorArrayV2Stack/TensorListStack?
#simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#simple_rnn_10/strided_slice_3/stack?
%simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%simple_rnn_10/strided_slice_3/stack_1?
%simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_3/stack_2?
simple_rnn_10/strided_slice_3StridedSlice9simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_10/strided_slice_3/stack:output:0.simple_rnn_10/strided_slice_3/stack_1:output:0.simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_10/strided_slice_3?
simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
simple_rnn_10/transpose_1/perm?
simple_rnn_10/transpose_1	Transpose9simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_10/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
simple_rnn_10/transpose_1?
activation_25/ReluRelusimple_rnn_10/transpose_1:y:0*
T0*-
_output_shapes
:???????????2
activation_25/Relu?
dropout_10/IdentityIdentity activation_25/Relu:activations:0*
T0*-
_output_shapes
:???????????2
dropout_10/Identityv
simple_rnn_11/ShapeShapedropout_10/Identity:output:0*
T0*
_output_shapes
:2
simple_rnn_11/Shape?
!simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!simple_rnn_11/strided_slice/stack?
#simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_11/strided_slice/stack_1?
#simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_11/strided_slice/stack_2?
simple_rnn_11/strided_sliceStridedSlicesimple_rnn_11/Shape:output:0*simple_rnn_11/strided_slice/stack:output:0,simple_rnn_11/strided_slice/stack_1:output:0,simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_11/strided_slicex
simple_rnn_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_11/zeros/mul/y?
simple_rnn_11/zeros/mulMul$simple_rnn_11/strided_slice:output:0"simple_rnn_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_11/zeros/mul{
simple_rnn_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_11/zeros/Less/y?
simple_rnn_11/zeros/LessLesssimple_rnn_11/zeros/mul:z:0#simple_rnn_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_11/zeros/Less~
simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_11/zeros/packed/1?
simple_rnn_11/zeros/packedPack$simple_rnn_11/strided_slice:output:0%simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_11/zeros/packed{
simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_11/zeros/Const?
simple_rnn_11/zerosFill#simple_rnn_11/zeros/packed:output:0"simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
simple_rnn_11/zeros?
simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_11/transpose/perm?
simple_rnn_11/transpose	Transposedropout_10/Identity:output:0%simple_rnn_11/transpose/perm:output:0*
T0*-
_output_shapes
:???????????2
simple_rnn_11/transposey
simple_rnn_11/Shape_1Shapesimple_rnn_11/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_11/Shape_1?
#simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_11/strided_slice_1/stack?
%simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_1/stack_1?
%simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_1/stack_2?
simple_rnn_11/strided_slice_1StridedSlicesimple_rnn_11/Shape_1:output:0,simple_rnn_11/strided_slice_1/stack:output:0.simple_rnn_11/strided_slice_1/stack_1:output:0.simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_11/strided_slice_1?
)simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)simple_rnn_11/TensorArrayV2/element_shape?
simple_rnn_11/TensorArrayV2TensorListReserve2simple_rnn_11/TensorArrayV2/element_shape:output:0&simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_11/TensorArrayV2?
Csimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2E
Csimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape?
5simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_11/transpose:y:0Lsimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5simple_rnn_11/TensorArrayUnstack/TensorListFromTensor?
#simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_11/strided_slice_2/stack?
%simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_2/stack_1?
%simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_2/stack_2?
simple_rnn_11/strided_slice_2StridedSlicesimple_rnn_11/transpose:y:0,simple_rnn_11/strided_slice_2/stack:output:0.simple_rnn_11/strided_slice_2/stack_1:output:0.simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_11/strided_slice_2?
6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp?simple_rnn_11_simple_rnn_cell_21_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype028
6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp?
'simple_rnn_11/simple_rnn_cell_21/MatMul	MLCMatMul&simple_rnn_11/strided_slice_2:output:0>simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22)
'simple_rnn_11/simple_rnn_cell_21/MatMul?
7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_11_simple_rnn_cell_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype029
7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
(simple_rnn_11/simple_rnn_cell_21/BiasAddBiasAdd1simple_rnn_11/simple_rnn_cell_21/MatMul:product:0?simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22*
(simple_rnn_11/simple_rnn_cell_21/BiasAdd?
8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_11_simple_rnn_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02:
8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
)simple_rnn_11/simple_rnn_cell_21/MatMul_1	MLCMatMulsimple_rnn_11/zeros:output:0@simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)simple_rnn_11/simple_rnn_cell_21/MatMul_1?
$simple_rnn_11/simple_rnn_cell_21/addAddV21simple_rnn_11/simple_rnn_cell_21/BiasAdd:output:03simple_rnn_11/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22&
$simple_rnn_11/simple_rnn_cell_21/add?
%simple_rnn_11/simple_rnn_cell_21/TanhTanh(simple_rnn_11/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22'
%simple_rnn_11/simple_rnn_cell_21/Tanh?
+simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2-
+simple_rnn_11/TensorArrayV2_1/element_shape?
simple_rnn_11/TensorArrayV2_1TensorListReserve4simple_rnn_11/TensorArrayV2_1/element_shape:output:0&simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_11/TensorArrayV2_1j
simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_11/time?
&simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&simple_rnn_11/while/maximum_iterations?
 simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 simple_rnn_11/while/loop_counter?
simple_rnn_11/whileWhile)simple_rnn_11/while/loop_counter:output:0/simple_rnn_11/while/maximum_iterations:output:0simple_rnn_11/time:output:0&simple_rnn_11/TensorArrayV2_1:handle:0simple_rnn_11/zeros:output:0&simple_rnn_11/strided_slice_1:output:0Esimple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_11_simple_rnn_cell_21_matmul_readvariableop_resource@simple_rnn_11_simple_rnn_cell_21_biasadd_readvariableop_resourceAsimple_rnn_11_simple_rnn_cell_21_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*+
body#R!
simple_rnn_11_while_body_242968*+
cond#R!
simple_rnn_11_while_cond_242967*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
simple_rnn_11/while?
>simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape?
0simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_11/while:output:3Gsimple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype022
0simple_rnn_11/TensorArrayV2Stack/TensorListStack?
#simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#simple_rnn_11/strided_slice_3/stack?
%simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%simple_rnn_11/strided_slice_3/stack_1?
%simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_3/stack_2?
simple_rnn_11/strided_slice_3StridedSlice9simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_11/strided_slice_3/stack:output:0.simple_rnn_11/strided_slice_3/stack_1:output:0.simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
simple_rnn_11/strided_slice_3?
simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
simple_rnn_11/transpose_1/perm?
simple_rnn_11/transpose_1	Transpose9simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_11/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22
simple_rnn_11/transpose_1?
activation_26/ReluRelu&simple_rnn_11/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22
activation_26/Relu?
dropout_11/IdentityIdentity activation_26/Relu:activations:0*
T0*'
_output_shapes
:?????????22
dropout_11/Identity?
!dense_35/MLCMatMul/ReadVariableOpReadVariableOp*dense_35_mlcmatmul_readvariableop_resource*
_output_shapes

:2H*
dtype02#
!dense_35/MLCMatMul/ReadVariableOp?
dense_35/MLCMatMul	MLCMatMuldropout_11/Identity:output:0)dense_35/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_35/MLCMatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MLCMatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_35/BiasAdds
dense_35/TanhTanhdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_35/Tanh?
IdentityIdentitydense_35/Tanh:y:0 ^dense_35/BiasAdd/ReadVariableOp"^dense_35/MLCMatMul/ReadVariableOp8^simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp7^simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp9^simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp^simple_rnn_10/while8^simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp7^simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp9^simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp^simple_rnn_11/while*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2F
!dense_35/MLCMatMul/ReadVariableOp!dense_35/MLCMatMul/ReadVariableOp2r
7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp2p
6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp2t
8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp2*
simple_rnn_10/whilesimple_rnn_10/while2r
7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp2p
6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp2t
8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp2*
simple_rnn_11/whilesimple_rnn_11/while:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_11_layer_call_fn_244137

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2423772
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?=
?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_241162

inputs
simple_rnn_cell_20_241087
simple_rnn_cell_20_241089
simple_rnn_cell_20_241091
identity??*simple_rnn_cell_20/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
*simple_rnn_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_20_241087simple_rnn_cell_20_241089simple_rnn_cell_20_241091*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_2408252,
*simple_rnn_cell_20/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_20_241087simple_rnn_cell_20_241089simple_rnn_cell_20_241091*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_241099*
condR
while_cond_241098*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0+^simple_rnn_cell_20/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2X
*simple_rnn_cell_20/StatefulPartitionedCall*simple_rnn_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_241610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241610___redundant_placeholder04
0while_while_cond_241610___redundant_placeholder14
0while_while_cond_241610___redundant_placeholder24
0while_while_cond_241610___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?H
?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_242320

inputs5
1simple_rnn_cell_21_matmul_readvariableop_resource6
2simple_rnn_cell_21_biasadd_readvariableop_resource7
3simple_rnn_cell_21_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_21/BiasAdd/ReadVariableOp?(simple_rnn_cell_21/MatMul/ReadVariableOp?*simple_rnn_cell_21/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:???????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_21_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_21/MatMul/ReadVariableOp?
simple_rnn_cell_21/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul?
)simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_21/BiasAdd/ReadVariableOp?
simple_rnn_cell_21/BiasAddBiasAdd#simple_rnn_cell_21/MatMul:product:01simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/BiasAdd?
*simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_21/MatMul_1/ReadVariableOp?
simple_rnn_cell_21/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul_1?
simple_rnn_cell_21/addAddV2#simple_rnn_cell_21/BiasAdd:output:0%simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/add?
simple_rnn_cell_21/TanhTanhsimple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_21_matmul_readvariableop_resource2simple_rnn_cell_21_biasadd_readvariableop_resource3simple_rnn_cell_21_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_242254*
condR
while_cond_242253*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_21/BiasAdd/ReadVariableOp)^simple_rnn_cell_21/MatMul/ReadVariableOp+^simple_rnn_cell_21/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::2V
)simple_rnn_cell_21/BiasAdd/ReadVariableOp)simple_rnn_cell_21/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_21/MatMul/ReadVariableOp(simple_rnn_cell_21/MatMul/ReadVariableOp2X
*simple_rnn_cell_21/MatMul_1/ReadVariableOp*simple_rnn_cell_21/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_30_layer_call_and_return_conditional_losses_242445
simple_rnn_10_input
simple_rnn_10_242421
simple_rnn_10_242423
simple_rnn_10_242425
simple_rnn_11_242430
simple_rnn_11_242432
simple_rnn_11_242434
dense_35_242439
dense_35_242441
identity?? dense_35/StatefulPartitionedCall?%simple_rnn_10/StatefulPartitionedCall?%simple_rnn_11/StatefulPartitionedCall?
%simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_10_inputsimple_rnn_10_242421simple_rnn_10_242423simple_rnn_10_242425*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_2420272'
%simple_rnn_10/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall.simple_rnn_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_2420622
activation_25/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall&activation_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2420842
dropout_10/PartitionedCall?
%simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0simple_rnn_11_242430simple_rnn_11_242432simple_rnn_11_242434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_2423202'
%simple_rnn_11/StatefulPartitionedCall?
activation_26/PartitionedCallPartitionedCall.simple_rnn_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_26_layer_call_and_return_conditional_losses_2423552
activation_26/PartitionedCall?
dropout_11/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2423772
dropout_11/PartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_35_242439dense_35_242441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_2424012"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_35/StatefulPartitionedCall&^simple_rnn_10/StatefulPartitionedCall&^simple_rnn_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%simple_rnn_10/StatefulPartitionedCall%simple_rnn_10/StatefulPartitionedCall2N
%simple_rnn_11/StatefulPartitionedCall%simple_rnn_11/StatefulPartitionedCall:a ]
,
_output_shapes
:??????????
-
_user_specified_namesimple_rnn_10_input
?H
?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243197
inputs_05
1simple_rnn_cell_20_matmul_readvariableop_resource6
2simple_rnn_cell_20_biasadd_readvariableop_resource7
3simple_rnn_cell_20_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_20/BiasAdd/ReadVariableOp?(simple_rnn_cell_20/MatMul/ReadVariableOp?*simple_rnn_cell_20/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_20/MatMul/ReadVariableOp?
simple_rnn_cell_20/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul?
)simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_20/BiasAdd/ReadVariableOp?
simple_rnn_cell_20/BiasAddBiasAdd#simple_rnn_cell_20/MatMul:product:01simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/BiasAdd?
*simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_20/MatMul_1/ReadVariableOp?
simple_rnn_cell_20/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul_1?
simple_rnn_cell_20/addAddV2#simple_rnn_cell_20/BiasAdd:output:0%simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/add?
simple_rnn_cell_20/TanhTanhsimple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_20_matmul_readvariableop_resource2simple_rnn_cell_20_biasadd_readvariableop_resource3simple_rnn_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_243131*
condR
while_cond_243130*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_20/BiasAdd/ReadVariableOp)^simple_rnn_cell_20/MatMul/ReadVariableOp+^simple_rnn_cell_20/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_20/BiasAdd/ReadVariableOp)simple_rnn_cell_20/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_20/MatMul/ReadVariableOp(simple_rnn_cell_20/MatMul/ReadVariableOp2X
*simple_rnn_cell_20/MatMul_1/ReadVariableOp*simple_rnn_cell_20/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_243601

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
3__inference_simple_rnn_cell_21_layer_call_fn_244281

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_2413542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?4
?
while_body_241961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_20_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_20_matmul_readvariableop_resource<
8while_simple_rnn_cell_20_biasadd_readvariableop_resource=
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource??/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_20/MatMul/ReadVariableOp?0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_20/MatMul/ReadVariableOp?
while/simple_rnn_cell_20/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_20/MatMul?
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_20/BiasAddBiasAdd)while/simple_rnn_cell_20/MatMul:product:07while/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_20/BiasAdd?
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_20/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_20/MatMul_1?
while/simple_rnn_cell_20/addAddV2)while/simple_rnn_cell_20/BiasAdd:output:0+while/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/add?
while/simple_rnn_cell_20/TanhTanh while/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_20/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_20/Tanh:y:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_20_biasadd_readvariableop_resource:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_20_matmul_readvariableop_resource9while_simple_rnn_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_20/MatMul/ReadVariableOp.while/simple_rnn_cell_20/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?4
?
while_body_243769
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_21_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_21_matmul_readvariableop_resource<
8while_simple_rnn_cell_21_biasadd_readvariableop_resource=
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource??/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_21/MatMul/ReadVariableOp?0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_21_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_21/MatMul/ReadVariableOp?
while/simple_rnn_cell_21/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_21/MatMul?
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_21/BiasAddBiasAdd)while/simple_rnn_cell_21/MatMul:product:07while/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_21/BiasAdd?
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_21/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_21/MatMul_1?
while/simple_rnn_cell_21/addAddV2)while/simple_rnn_cell_21/BiasAdd:output:0+while/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/add?
while/simple_rnn_cell_21/TanhTanh while/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_21/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_21/Tanh:y:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_21_biasadd_readvariableop_resource:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_21_matmul_readvariableop_resource9while_simple_rnn_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_21/MatMul/ReadVariableOp.while/simple_rnn_cell_21/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_signature_wrapper_242573
simple_rnn_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_2407762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
,
_output_shapes
:??????????
-
_user_specified_namesimple_rnn_10_input
?
b
F__inference_dropout_10_layer_call_and_return_conditional_losses_243596

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consty
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????2
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*-
_output_shapes
:???????????2	
dropoutj
IdentityIdentitydropout:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_244174

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpw
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp
MatMul_1	MLCMatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
e
I__inference_activation_25_layer_call_and_return_conditional_losses_243582

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
.__inference_simple_rnn_10_layer_call_fn_243577

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_2420272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?4
?
while_body_242254
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_21_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_21_matmul_readvariableop_resource<
8while_simple_rnn_cell_21_biasadd_readvariableop_resource=
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource??/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_21/MatMul/ReadVariableOp?0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_21_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_21/MatMul/ReadVariableOp?
while/simple_rnn_cell_21/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_21/MatMul?
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_21/BiasAddBiasAdd)while/simple_rnn_cell_21/MatMul:product:07while/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_21/BiasAdd?
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_21/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_21/MatMul_1?
while/simple_rnn_cell_21/addAddV2)while/simple_rnn_cell_21/BiasAdd:output:0+while/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/add?
while/simple_rnn_cell_21/TanhTanh while/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_21/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_21/Tanh:y:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_21_biasadd_readvariableop_resource:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_21_matmul_readvariableop_resource9while_simple_rnn_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_21/MatMul/ReadVariableOp.while/simple_rnn_cell_21/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_simple_rnn_11_layer_call_fn_244103
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_2417912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
J
.__inference_activation_26_layer_call_fn_244113

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_26_layer_call_and_return_conditional_losses_2423552
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?H
?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_242027

inputs5
1simple_rnn_cell_20_matmul_readvariableop_resource6
2simple_rnn_cell_20_biasadd_readvariableop_resource7
3simple_rnn_cell_20_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_20/BiasAdd/ReadVariableOp?(simple_rnn_cell_20/MatMul/ReadVariableOp?*simple_rnn_cell_20/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_20/MatMul/ReadVariableOp?
simple_rnn_cell_20/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul?
)simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_20/BiasAdd/ReadVariableOp?
simple_rnn_cell_20/BiasAddBiasAdd#simple_rnn_cell_20/MatMul:product:01simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/BiasAdd?
*simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_20/MatMul_1/ReadVariableOp?
simple_rnn_cell_20/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul_1?
simple_rnn_cell_20/addAddV2#simple_rnn_cell_20/BiasAdd:output:0%simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/add?
simple_rnn_cell_20/TanhTanhsimple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_20_matmul_readvariableop_resource2simple_rnn_cell_20_biasadd_readvariableop_resource3simple_rnn_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_241961*
condR
while_cond_241960*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_20/BiasAdd/ReadVariableOp)^simple_rnn_cell_20/MatMul/ReadVariableOp+^simple_rnn_cell_20/MatMul_1/ReadVariableOp^while*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::2V
)simple_rnn_cell_20/BiasAdd/ReadVariableOp)simple_rnn_cell_20/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_20/MatMul/ReadVariableOp(simple_rnn_cell_20/MatMul/ReadVariableOp2X
*simple_rnn_cell_20/MatMul_1/ReadVariableOp*simple_rnn_cell_20/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_244014
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_244014___redundant_placeholder04
0while_while_cond_244014___redundant_placeholder14
0while_while_cond_244014___redundant_placeholder24
0while_while_cond_244014___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_241337

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOpv
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1	MLCMatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????22
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?#
?
while_body_241099
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_20_241121_0%
!while_simple_rnn_cell_20_241123_0%
!while_simple_rnn_cell_20_241125_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_20_241121#
while_simple_rnn_cell_20_241123#
while_simple_rnn_cell_20_241125??0while/simple_rnn_cell_20/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
0while/simple_rnn_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_20_241121_0!while_simple_rnn_cell_20_241123_0!while_simple_rnn_cell_20_241125_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_24082522
0while/simple_rnn_cell_20/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_20/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/simple_rnn_cell_20/StatefulPartitionedCall:output:11^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_20_241121!while_simple_rnn_cell_20_241121_0"D
while_simple_rnn_cell_20_241123!while_simple_rnn_cell_20_241123_0"D
while_simple_rnn_cell_20_241125!while_simple_rnn_cell_20_241125_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2d
0while/simple_rnn_cell_20/StatefulPartitionedCall0while/simple_rnn_cell_20/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_sequential_30_layer_call_fn_242542
simple_rnn_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_30_layer_call_and_return_conditional_losses_2425232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
,
_output_shapes
:??????????
-
_user_specified_namesimple_rnn_10_input
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_242084

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?<
?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_241674

inputs
simple_rnn_cell_21_241599
simple_rnn_cell_21_241601
simple_rnn_cell_21_241603
identity??*simple_rnn_cell_21/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
*simple_rnn_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_21_241599simple_rnn_cell_21_241601simple_rnn_cell_21_241603*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_2413372,
*simple_rnn_cell_21/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_21_241599simple_rnn_cell_21_241601simple_rnn_cell_21_241603*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_241611*
condR
while_cond_241610*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0+^simple_rnn_cell_21/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2X
*simple_rnn_cell_21/StatefulPartitionedCall*simple_rnn_cell_21/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
while_cond_241098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241098___redundant_placeholder04
0while_while_cond_241098___redundant_placeholder14
0while_while_cond_241098___redundant_placeholder24
0while_while_cond_241098___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?4
?
while_body_241849
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_20_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_20_matmul_readvariableop_resource<
8while_simple_rnn_cell_20_biasadd_readvariableop_resource=
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource??/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_20/MatMul/ReadVariableOp?0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_20/MatMul/ReadVariableOp?
while/simple_rnn_cell_20/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_20/MatMul?
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_20/BiasAddBiasAdd)while/simple_rnn_cell_20/MatMul:product:07while/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_20/BiasAdd?
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_20/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_20/MatMul_1?
while/simple_rnn_cell_20/addAddV2)while/simple_rnn_cell_20/BiasAdd:output:0+while/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/add?
while/simple_rnn_cell_20/TanhTanh while/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_20/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_20/Tanh:y:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_20_biasadd_readvariableop_resource:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_20_matmul_readvariableop_resource9while_simple_rnn_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_20/MatMul/ReadVariableOp.while/simple_rnn_cell_20/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
simple_rnn_11_while_cond_2429678
4simple_rnn_11_while_simple_rnn_11_while_loop_counter>
:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations#
simple_rnn_11_while_placeholder%
!simple_rnn_11_while_placeholder_1%
!simple_rnn_11_while_placeholder_2:
6simple_rnn_11_while_less_simple_rnn_11_strided_slice_1P
Lsimple_rnn_11_while_simple_rnn_11_while_cond_242967___redundant_placeholder0P
Lsimple_rnn_11_while_simple_rnn_11_while_cond_242967___redundant_placeholder1P
Lsimple_rnn_11_while_simple_rnn_11_while_cond_242967___redundant_placeholder2P
Lsimple_rnn_11_while_simple_rnn_11_while_cond_242967___redundant_placeholder3 
simple_rnn_11_while_identity
?
simple_rnn_11/while/LessLesssimple_rnn_11_while_placeholder6simple_rnn_11_while_less_simple_rnn_11_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_11/while/Less?
simple_rnn_11/while/IdentityIdentitysimple_rnn_11/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_11/while/Identity"E
simple_rnn_11_while_identity%simple_rnn_11/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?4
?
while_body_243657
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_21_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_21_matmul_readvariableop_resource<
8while_simple_rnn_cell_21_biasadd_readvariableop_resource=
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource??/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_21/MatMul/ReadVariableOp?0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_21_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_21/MatMul/ReadVariableOp?
while/simple_rnn_cell_21/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_21/MatMul?
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_21/BiasAddBiasAdd)while/simple_rnn_cell_21/MatMul:product:07while/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_21/BiasAdd?
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_21/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_21/MatMul_1?
while/simple_rnn_cell_21/addAddV2)while/simple_rnn_cell_21/BiasAdd:output:0+while/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/add?
while/simple_rnn_cell_21/TanhTanh while/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_21/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_21/Tanh:y:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_21_biasadd_readvariableop_resource:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_21_matmul_readvariableop_resource9while_simple_rnn_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_21/MatMul/ReadVariableOp.while/simple_rnn_cell_21/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?H
?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_243969
inputs_05
1simple_rnn_cell_21_matmul_readvariableop_resource6
2simple_rnn_cell_21_biasadd_readvariableop_resource7
3simple_rnn_cell_21_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_21/BiasAdd/ReadVariableOp?(simple_rnn_cell_21/MatMul/ReadVariableOp?*simple_rnn_cell_21/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_21_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_21/MatMul/ReadVariableOp?
simple_rnn_cell_21/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul?
)simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_21/BiasAdd/ReadVariableOp?
simple_rnn_cell_21/BiasAddBiasAdd#simple_rnn_cell_21/MatMul:product:01simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/BiasAdd?
*simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_21/MatMul_1/ReadVariableOp?
simple_rnn_cell_21/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul_1?
simple_rnn_cell_21/addAddV2#simple_rnn_cell_21/BiasAdd:output:0%simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/add?
simple_rnn_cell_21/TanhTanhsimple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_21_matmul_readvariableop_resource2simple_rnn_cell_21_biasadd_readvariableop_resource3simple_rnn_cell_21_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_243903*
condR
while_cond_243902*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_21/BiasAdd/ReadVariableOp)^simple_rnn_cell_21/MatMul/ReadVariableOp+^simple_rnn_cell_21/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2V
)simple_rnn_cell_21/BiasAdd/ReadVariableOp)simple_rnn_cell_21/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_21/MatMul/ReadVariableOp(simple_rnn_cell_21/MatMul/ReadVariableOp2X
*simple_rnn_cell_21/MatMul_1/ReadVariableOp*simple_rnn_cell_21/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_243656
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_243656___redundant_placeholder04
0while_while_cond_243656___redundant_placeholder14
0while_while_cond_243656___redundant_placeholder24
0while_while_cond_243656___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?T
?
-sequential_30_simple_rnn_10_while_body_240591T
Psequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_loop_counterZ
Vsequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_maximum_iterations1
-sequential_30_simple_rnn_10_while_placeholder3
/sequential_30_simple_rnn_10_while_placeholder_13
/sequential_30_simple_rnn_10_while_placeholder_2S
Osequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_strided_slice_1_0?
?sequential_30_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0Y
Usequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resource_0Z
Vsequential_30_simple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resource_0[
Wsequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0.
*sequential_30_simple_rnn_10_while_identity0
,sequential_30_simple_rnn_10_while_identity_10
,sequential_30_simple_rnn_10_while_identity_20
,sequential_30_simple_rnn_10_while_identity_30
,sequential_30_simple_rnn_10_while_identity_4Q
Msequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_strided_slice_1?
?sequential_30_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorW
Ssequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resourceX
Tsequential_30_simple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resourceY
Usequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource??Ksequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?Jsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?Lsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
Ssequential_30/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2U
Ssequential_30/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Esequential_30/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_30_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0-sequential_30_simple_rnn_10_while_placeholder\sequential_30/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02G
Esequential_30/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem?
Jsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOpUsequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02L
Jsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?
;sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul	MLCMatMulLsequential_30/simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2=
;sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul?
Ksequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOpVsequential_30_simple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02M
Ksequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
<sequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAddBiasAddEsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul:product:0Ssequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2>
<sequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd?
Lsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOpWsequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02N
Lsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
=sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1	MLCMatMul/sequential_30_simple_rnn_10_while_placeholder_2Tsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2?
=sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1?
8sequential_30/simple_rnn_10/while/simple_rnn_cell_20/addAddV2Esequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd:output:0Gsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2:
8sequential_30/simple_rnn_10/while/simple_rnn_cell_20/add?
9sequential_30/simple_rnn_10/while/simple_rnn_cell_20/TanhTanh<sequential_30/simple_rnn_10/while/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2;
9sequential_30/simple_rnn_10/while/simple_rnn_cell_20/Tanh?
Fsequential_30/simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_30_simple_rnn_10_while_placeholder_1-sequential_30_simple_rnn_10_while_placeholder=sequential_30/simple_rnn_10/while/simple_rnn_cell_20/Tanh:y:0*
_output_shapes
: *
element_dtype02H
Fsequential_30/simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem?
'sequential_30/simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_30/simple_rnn_10/while/add/y?
%sequential_30/simple_rnn_10/while/addAddV2-sequential_30_simple_rnn_10_while_placeholder0sequential_30/simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: 2'
%sequential_30/simple_rnn_10/while/add?
)sequential_30/simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_30/simple_rnn_10/while/add_1/y?
'sequential_30/simple_rnn_10/while/add_1AddV2Psequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_loop_counter2sequential_30/simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2)
'sequential_30/simple_rnn_10/while/add_1?
*sequential_30/simple_rnn_10/while/IdentityIdentity+sequential_30/simple_rnn_10/while/add_1:z:0L^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOpM^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_30/simple_rnn_10/while/Identity?
,sequential_30/simple_rnn_10/while/Identity_1IdentityVsequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_maximum_iterationsL^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOpM^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2.
,sequential_30/simple_rnn_10/while/Identity_1?
,sequential_30/simple_rnn_10/while/Identity_2Identity)sequential_30/simple_rnn_10/while/add:z:0L^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOpM^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2.
,sequential_30/simple_rnn_10/while/Identity_2?
,sequential_30/simple_rnn_10/while/Identity_3IdentityVsequential_30/simple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0L^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOpM^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2.
,sequential_30/simple_rnn_10/while/Identity_3?
,sequential_30/simple_rnn_10/while/Identity_4Identity=sequential_30/simple_rnn_10/while/simple_rnn_cell_20/Tanh:y:0L^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpK^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOpM^sequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2.
,sequential_30/simple_rnn_10/while/Identity_4"a
*sequential_30_simple_rnn_10_while_identity3sequential_30/simple_rnn_10/while/Identity:output:0"e
,sequential_30_simple_rnn_10_while_identity_15sequential_30/simple_rnn_10/while/Identity_1:output:0"e
,sequential_30_simple_rnn_10_while_identity_25sequential_30/simple_rnn_10/while/Identity_2:output:0"e
,sequential_30_simple_rnn_10_while_identity_35sequential_30/simple_rnn_10/while/Identity_3:output:0"e
,sequential_30_simple_rnn_10_while_identity_45sequential_30/simple_rnn_10/while/Identity_4:output:0"?
Msequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_strided_slice_1Osequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_strided_slice_1_0"?
Tsequential_30_simple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resourceVsequential_30_simple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resource_0"?
Usequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resourceWsequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0"?
Ssequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resourceUsequential_30_simple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resource_0"?
?sequential_30_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor?sequential_30_simple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_sequential_30_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2?
Ksequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpKsequential_30/simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp2?
Jsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOpJsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp2?
Lsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOpLsequential_30/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?	
?
3__inference_simple_rnn_cell_20_layer_call_fn_244205

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_2408252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?D
?
simple_rnn_11_while_body_2427338
4simple_rnn_11_while_simple_rnn_11_while_loop_counter>
:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations#
simple_rnn_11_while_placeholder%
!simple_rnn_11_while_placeholder_1%
!simple_rnn_11_while_placeholder_27
3simple_rnn_11_while_simple_rnn_11_strided_slice_1_0s
osimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0K
Gsimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resource_0L
Hsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resource_0M
Isimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0 
simple_rnn_11_while_identity"
simple_rnn_11_while_identity_1"
simple_rnn_11_while_identity_2"
simple_rnn_11_while_identity_3"
simple_rnn_11_while_identity_45
1simple_rnn_11_while_simple_rnn_11_strided_slice_1q
msimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorI
Esimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resourceJ
Fsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resourceK
Gsimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource??=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
Esimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2G
Esimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape?
7simple_rnn_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_11_while_placeholderNsimple_rnn_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype029
7simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem?
<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02>
<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?
-simple_rnn_11/while/simple_rnn_cell_21/MatMul	MLCMatMul>simple_rnn_11/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22/
-simple_rnn_11/while/simple_rnn_cell_21/MatMul?
=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02?
=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
.simple_rnn_11/while/simple_rnn_cell_21/BiasAddBiasAdd7simple_rnn_11/while/simple_rnn_cell_21/MatMul:product:0Esimple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.simple_rnn_11/while/simple_rnn_cell_21/BiasAdd?
>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02@
>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1	MLCMatMul!simple_rnn_11_while_placeholder_2Fsimple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????221
/simple_rnn_11/while/simple_rnn_cell_21/MatMul_1?
*simple_rnn_11/while/simple_rnn_cell_21/addAddV27simple_rnn_11/while/simple_rnn_cell_21/BiasAdd:output:09simple_rnn_11/while/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22,
*simple_rnn_11/while/simple_rnn_cell_21/add?
+simple_rnn_11/while/simple_rnn_cell_21/TanhTanh.simple_rnn_11/while/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22-
+simple_rnn_11/while/simple_rnn_cell_21/Tanh?
8simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_11_while_placeholder_1simple_rnn_11_while_placeholder/simple_rnn_11/while/simple_rnn_cell_21/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8simple_rnn_11/while/TensorArrayV2Write/TensorListSetItemx
simple_rnn_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_11/while/add/y?
simple_rnn_11/while/addAddV2simple_rnn_11_while_placeholder"simple_rnn_11/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_11/while/add|
simple_rnn_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_11/while/add_1/y?
simple_rnn_11/while/add_1AddV24simple_rnn_11_while_simple_rnn_11_while_loop_counter$simple_rnn_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_11/while/add_1?
simple_rnn_11/while/IdentityIdentitysimple_rnn_11/while/add_1:z:0>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_11/while/Identity?
simple_rnn_11/while/Identity_1Identity:simple_rnn_11_while_simple_rnn_11_while_maximum_iterations>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_11/while/Identity_1?
simple_rnn_11/while/Identity_2Identitysimple_rnn_11/while/add:z:0>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_11/while/Identity_2?
simple_rnn_11/while/Identity_3IdentityHsimple_rnn_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_11/while/Identity_3?
simple_rnn_11/while/Identity_4Identity/simple_rnn_11/while/simple_rnn_cell_21/Tanh:y:0>^simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=^simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp?^simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22 
simple_rnn_11/while/Identity_4"E
simple_rnn_11_while_identity%simple_rnn_11/while/Identity:output:0"I
simple_rnn_11_while_identity_1'simple_rnn_11/while/Identity_1:output:0"I
simple_rnn_11_while_identity_2'simple_rnn_11/while/Identity_2:output:0"I
simple_rnn_11_while_identity_3'simple_rnn_11/while/Identity_3:output:0"I
simple_rnn_11_while_identity_4'simple_rnn_11/while/Identity_4:output:0"h
1simple_rnn_11_while_simple_rnn_11_strided_slice_13simple_rnn_11_while_simple_rnn_11_strided_slice_1_0"?
Fsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resourceHsimple_rnn_11_while_simple_rnn_cell_21_biasadd_readvariableop_resource_0"?
Gsimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resourceIsimple_rnn_11_while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0"?
Esimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resourceGsimple_rnn_11_while_simple_rnn_cell_21_matmul_readvariableop_resource_0"?
msimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensorosimple_rnn_11_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_11_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2~
=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp=simple_rnn_11/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp2|
<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp<simple_rnn_11/while/simple_rnn_cell_21/MatMul/ReadVariableOp2?
>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp>simple_rnn_11/while/simple_rnn_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_dense_35_layer_call_and_return_conditional_losses_242401

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:2H*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?H
?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_243835

inputs5
1simple_rnn_cell_21_matmul_readvariableop_resource6
2simple_rnn_cell_21_biasadd_readvariableop_resource7
3simple_rnn_cell_21_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_21/BiasAdd/ReadVariableOp?(simple_rnn_cell_21/MatMul/ReadVariableOp?*simple_rnn_cell_21/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:???????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_21_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_21/MatMul/ReadVariableOp?
simple_rnn_cell_21/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul?
)simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_21/BiasAdd/ReadVariableOp?
simple_rnn_cell_21/BiasAddBiasAdd#simple_rnn_cell_21/MatMul:product:01simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/BiasAdd?
*simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_21/MatMul_1/ReadVariableOp?
simple_rnn_cell_21/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul_1?
simple_rnn_cell_21/addAddV2#simple_rnn_cell_21/BiasAdd:output:0%simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/add?
simple_rnn_cell_21/TanhTanhsimple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_21_matmul_readvariableop_resource2simple_rnn_cell_21_biasadd_readvariableop_resource3simple_rnn_cell_21_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_243769*
condR
while_cond_243768*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_21/BiasAdd/ReadVariableOp)^simple_rnn_cell_21/MatMul/ReadVariableOp+^simple_rnn_cell_21/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::2V
)simple_rnn_cell_21/BiasAdd/ReadVariableOp)simple_rnn_cell_21/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_21/MatMul/ReadVariableOp(simple_rnn_cell_21/MatMul/ReadVariableOp2X
*simple_rnn_cell_21/MatMul_1/ReadVariableOp*simple_rnn_cell_21/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_240842

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpw
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp}
MatMul_1	MLCMatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_240825

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpw
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp}
MatMul_1	MLCMatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
addP
TanhTanhadd:z:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
while_cond_243902
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_243902___redundant_placeholder04
0while_while_cond_243902___redundant_placeholder14
0while_while_cond_243902___redundant_placeholder24
0while_while_cond_243902___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
b
F__inference_dropout_10_layer_call_and_return_conditional_losses_242079

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consty
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????2
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*-
_output_shapes
:???????????2	
dropoutj
IdentityIdentitydropout:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_244236

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOpv
MatMul	MLCMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul_1/ReadVariableOp~
MatMul_1	MLCMatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????22
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?4
?
while_body_243489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_20_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_20_matmul_readvariableop_resource<
8while_simple_rnn_cell_20_biasadd_readvariableop_resource=
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource??/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_20/MatMul/ReadVariableOp?0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_20/MatMul/ReadVariableOp?
while/simple_rnn_cell_20/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_20/MatMul?
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_20/BiasAddBiasAdd)while/simple_rnn_cell_20/MatMul:product:07while/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_20/BiasAdd?
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_20/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_20/MatMul_1?
while/simple_rnn_cell_20/addAddV2)while/simple_rnn_cell_20/BiasAdd:output:0+while/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/add?
while/simple_rnn_cell_20/TanhTanh while/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_20/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_20/Tanh:y:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_20_biasadd_readvariableop_resource:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_20_matmul_readvariableop_resource9while_simple_rnn_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_20/MatMul/ReadVariableOp.while/simple_rnn_cell_20/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_sequential_30_layer_call_fn_242494
simple_rnn_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_30_layer_call_and_return_conditional_losses_2424752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
,
_output_shapes
:??????????
-
_user_specified_namesimple_rnn_10_input
?#
?
while_body_241216
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_20_241238_0%
!while_simple_rnn_cell_20_241240_0%
!while_simple_rnn_cell_20_241242_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_20_241238#
while_simple_rnn_cell_20_241240#
while_simple_rnn_cell_20_241242??0while/simple_rnn_cell_20/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
0while/simple_rnn_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_20_241238_0!while_simple_rnn_cell_20_241240_0!while_simple_rnn_cell_20_241242_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_24084222
0while/simple_rnn_cell_20/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_20/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/simple_rnn_cell_20/StatefulPartitionedCall:output:11^while/simple_rnn_cell_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_20_241238!while_simple_rnn_cell_20_241238_0"D
while_simple_rnn_cell_20_241240!while_simple_rnn_cell_20_241240_0"D
while_simple_rnn_cell_20_241242!while_simple_rnn_cell_20_241242_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2d
0while/simple_rnn_cell_20/StatefulPartitionedCall0while/simple_rnn_cell_20/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
simple_rnn_10_while_cond_2428578
4simple_rnn_10_while_simple_rnn_10_while_loop_counter>
:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations#
simple_rnn_10_while_placeholder%
!simple_rnn_10_while_placeholder_1%
!simple_rnn_10_while_placeholder_2:
6simple_rnn_10_while_less_simple_rnn_10_strided_slice_1P
Lsimple_rnn_10_while_simple_rnn_10_while_cond_242857___redundant_placeholder0P
Lsimple_rnn_10_while_simple_rnn_10_while_cond_242857___redundant_placeholder1P
Lsimple_rnn_10_while_simple_rnn_10_while_cond_242857___redundant_placeholder2P
Lsimple_rnn_10_while_simple_rnn_10_while_cond_242857___redundant_placeholder3 
simple_rnn_10_while_identity
?
simple_rnn_10/while/LessLesssimple_rnn_10_while_placeholder6simple_rnn_10_while_less_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_10/while/Less?
simple_rnn_10/while/IdentityIdentitysimple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_10/while/Identity"E
simple_rnn_10_while_identity%simple_rnn_10/while/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
J
.__inference_activation_25_layer_call_fn_243587

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_2420622
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?#
?
while_body_241728
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0%
!while_simple_rnn_cell_21_241750_0%
!while_simple_rnn_cell_21_241752_0%
!while_simple_rnn_cell_21_241754_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor#
while_simple_rnn_cell_21_241750#
while_simple_rnn_cell_21_241752#
while_simple_rnn_cell_21_241754??0while/simple_rnn_cell_21/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
0while/simple_rnn_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_21_241750_0!while_simple_rnn_cell_21_241752_0!while_simple_rnn_cell_21_241754_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_24135422
0while/simple_rnn_cell_21/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/simple_rnn_cell_21/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/simple_rnn_cell_21/StatefulPartitionedCall:output:11^while/simple_rnn_cell_21/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_21_241750!while_simple_rnn_cell_21_241750_0"D
while_simple_rnn_cell_21_241752!while_simple_rnn_cell_21_241752_0"D
while_simple_rnn_cell_21_241754!while_simple_rnn_cell_21_241754_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2d
0while/simple_rnn_cell_21/StatefulPartitionedCall0while/simple_rnn_cell_21/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?=
?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_241279

inputs
simple_rnn_cell_20_241204
simple_rnn_cell_20_241206
simple_rnn_cell_20_241208
identity??*simple_rnn_cell_20/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
*simple_rnn_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_20_241204simple_rnn_cell_20_241206simple_rnn_cell_20_241208*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_2408422,
*simple_rnn_cell_20/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_20_241204simple_rnn_cell_20_241206simple_rnn_cell_20_241208*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_241216*
condR
while_cond_241215*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitytranspose_1:y:0+^simple_rnn_cell_20/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2X
*simple_rnn_cell_20/StatefulPartitionedCall*simple_rnn_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?H
?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243443

inputs5
1simple_rnn_cell_20_matmul_readvariableop_resource6
2simple_rnn_cell_20_biasadd_readvariableop_resource7
3simple_rnn_cell_20_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_20/BiasAdd/ReadVariableOp?(simple_rnn_cell_20/MatMul/ReadVariableOp?*simple_rnn_cell_20/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_20/MatMul/ReadVariableOp?
simple_rnn_cell_20/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul?
)simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_20/BiasAdd/ReadVariableOp?
simple_rnn_cell_20/BiasAddBiasAdd#simple_rnn_cell_20/MatMul:product:01simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/BiasAdd?
*simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_20/MatMul_1/ReadVariableOp?
simple_rnn_cell_20/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul_1?
simple_rnn_cell_20/addAddV2#simple_rnn_cell_20/BiasAdd:output:0%simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/add?
simple_rnn_cell_20/TanhTanhsimple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_20_matmul_readvariableop_resource2simple_rnn_cell_20_biasadd_readvariableop_resource3simple_rnn_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_243377*
condR
while_cond_243376*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_20/BiasAdd/ReadVariableOp)^simple_rnn_cell_20/MatMul/ReadVariableOp+^simple_rnn_cell_20/MatMul_1/ReadVariableOp^while*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::2V
)simple_rnn_cell_20/BiasAdd/ReadVariableOp)simple_rnn_cell_20/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_20/MatMul/ReadVariableOp(simple_rnn_cell_20/MatMul/ReadVariableOp2X
*simple_rnn_cell_20/MatMul_1/ReadVariableOp*simple_rnn_cell_20/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?H
?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_243723

inputs5
1simple_rnn_cell_21_matmul_readvariableop_resource6
2simple_rnn_cell_21_biasadd_readvariableop_resource7
3simple_rnn_cell_21_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_21/BiasAdd/ReadVariableOp?(simple_rnn_cell_21/MatMul/ReadVariableOp?*simple_rnn_cell_21/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:???????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_21_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(simple_rnn_cell_21/MatMul/ReadVariableOp?
simple_rnn_cell_21/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul?
)simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)simple_rnn_cell_21/BiasAdd/ReadVariableOp?
simple_rnn_cell_21/BiasAddBiasAdd#simple_rnn_cell_21/MatMul:product:01simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/BiasAdd?
*simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02,
*simple_rnn_cell_21/MatMul_1/ReadVariableOp?
simple_rnn_cell_21/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/MatMul_1?
simple_rnn_cell_21/addAddV2#simple_rnn_cell_21/BiasAdd:output:0%simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/add?
simple_rnn_cell_21/TanhTanhsimple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_21/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_21_matmul_readvariableop_resource2simple_rnn_cell_21_biasadd_readvariableop_resource3simple_rnn_cell_21_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_243657*
condR
while_cond_243656*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_21/BiasAdd/ReadVariableOp)^simple_rnn_cell_21/MatMul/ReadVariableOp+^simple_rnn_cell_21/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::2V
)simple_rnn_cell_21/BiasAdd/ReadVariableOp)simple_rnn_cell_21/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_21/MatMul/ReadVariableOp(simple_rnn_cell_21/MatMul/ReadVariableOp2X
*simple_rnn_cell_21/MatMul_1/ReadVariableOp*simple_rnn_cell_21/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
while_cond_242253
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_242253___redundant_placeholder04
0while_while_cond_242253___redundant_placeholder14
0while_while_cond_242253___redundant_placeholder24
0while_while_cond_242253___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
-sequential_30_simple_rnn_10_while_cond_240590T
Psequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_loop_counterZ
Vsequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_maximum_iterations1
-sequential_30_simple_rnn_10_while_placeholder3
/sequential_30_simple_rnn_10_while_placeholder_13
/sequential_30_simple_rnn_10_while_placeholder_2V
Rsequential_30_simple_rnn_10_while_less_sequential_30_simple_rnn_10_strided_slice_1l
hsequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_cond_240590___redundant_placeholder0l
hsequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_cond_240590___redundant_placeholder1l
hsequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_cond_240590___redundant_placeholder2l
hsequential_30_simple_rnn_10_while_sequential_30_simple_rnn_10_while_cond_240590___redundant_placeholder3.
*sequential_30_simple_rnn_10_while_identity
?
&sequential_30/simple_rnn_10/while/LessLess-sequential_30_simple_rnn_10_while_placeholderRsequential_30_simple_rnn_10_while_less_sequential_30_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: 2(
&sequential_30/simple_rnn_10/while/Less?
*sequential_30/simple_rnn_10/while/IdentityIdentity*sequential_30/simple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: 2,
*sequential_30/simple_rnn_10/while/Identity"a
*sequential_30_simple_rnn_10_while_identity3sequential_30/simple_rnn_10/while/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
.__inference_simple_rnn_10_layer_call_fn_243566

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_2419152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
simple_rnn_10_while_cond_2426188
4simple_rnn_10_while_simple_rnn_10_while_loop_counter>
:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations#
simple_rnn_10_while_placeholder%
!simple_rnn_10_while_placeholder_1%
!simple_rnn_10_while_placeholder_2:
6simple_rnn_10_while_less_simple_rnn_10_strided_slice_1P
Lsimple_rnn_10_while_simple_rnn_10_while_cond_242618___redundant_placeholder0P
Lsimple_rnn_10_while_simple_rnn_10_while_cond_242618___redundant_placeholder1P
Lsimple_rnn_10_while_simple_rnn_10_while_cond_242618___redundant_placeholder2P
Lsimple_rnn_10_while_simple_rnn_10_while_cond_242618___redundant_placeholder3 
simple_rnn_10_while_identity
?
simple_rnn_10/while/LessLesssimple_rnn_10_while_placeholder6simple_rnn_10_while_less_simple_rnn_10_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_10/while/Less?
simple_rnn_10/while/IdentityIdentitysimple_rnn_10/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_10/while/Identity"E
simple_rnn_10_while_identity%simple_rnn_10/while/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?<
?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_241791

inputs
simple_rnn_cell_21_241716
simple_rnn_cell_21_241718
simple_rnn_cell_21_241720
identity??*simple_rnn_cell_21/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
*simple_rnn_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_21_241716simple_rnn_cell_21_241718simple_rnn_cell_21_241720*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_2413542,
*simple_rnn_cell_21/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_21_241716simple_rnn_cell_21_241718simple_rnn_cell_21_241720*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_241728*
condR
while_cond_241727*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0+^simple_rnn_cell_21/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2X
*simple_rnn_cell_21/StatefulPartitionedCall*simple_rnn_cell_21/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
while_cond_242141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_242141___redundant_placeholder04
0while_while_cond_242141___redundant_placeholder14
0while_while_cond_242141___redundant_placeholder24
0while_while_cond_242141___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_243130
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_243130___redundant_placeholder04
0while_while_cond_243130___redundant_placeholder14
0while_while_cond_243130___redundant_placeholder24
0while_while_cond_243130___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?4
?
while_body_243903
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_21_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_21_matmul_readvariableop_resource<
8while_simple_rnn_cell_21_biasadd_readvariableop_resource=
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource??/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_21/MatMul/ReadVariableOp?0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_21_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype020
.while/simple_rnn_cell_21/MatMul/ReadVariableOp?
while/simple_rnn_cell_21/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_21/MatMul?
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype021
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_21/BiasAddBiasAdd)while/simple_rnn_cell_21/MatMul:product:07while/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_21/BiasAdd?
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype022
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_21/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22#
!while/simple_rnn_cell_21/MatMul_1?
while/simple_rnn_cell_21/addAddV2)while/simple_rnn_cell_21/BiasAdd:output:0+while/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/add?
while/simple_rnn_cell_21/TanhTanh while/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_21/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_21/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_21/Tanh:y:00^while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_21/MatMul/ReadVariableOp1^while/simple_rnn_cell_21/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_21_biasadd_readvariableop_resource:while_simple_rnn_cell_21_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_21_matmul_1_readvariableop_resource;while_simple_rnn_cell_21_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_21_matmul_readvariableop_resource9while_simple_rnn_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp/while/simple_rnn_cell_21/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_21/MatMul/ReadVariableOp.while/simple_rnn_cell_21/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp0while/simple_rnn_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_242377

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
.__inference_simple_rnn_11_layer_call_fn_244092
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_2416742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_243488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_243488___redundant_placeholder04
0while_while_cond_243488___redundant_placeholder14
0while_while_cond_243488___redundant_placeholder24
0while_while_cond_243488___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
.__inference_simple_rnn_11_layer_call_fn_243846

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_2422082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_30_layer_call_fn_243064

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_30_layer_call_and_return_conditional_losses_2424752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?P
?
__inference__traced_save_244409
file_prefix.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopF
Bsavev2_simple_rnn_10_simple_rnn_cell_20_kernel_read_readvariableopP
Lsavev2_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_10_simple_rnn_cell_20_bias_read_readvariableopF
Bsavev2_simple_rnn_11_simple_rnn_cell_21_kernel_read_readvariableopP
Lsavev2_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_11_simple_rnn_cell_21_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_10_simple_rnn_cell_20_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_10_simple_rnn_cell_20_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_11_simple_rnn_cell_21_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_11_simple_rnn_cell_21_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_10_simple_rnn_cell_20_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_10_simple_rnn_cell_20_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_11_simple_rnn_cell_21_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_11_simple_rnn_cell_21_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopBsavev2_simple_rnn_10_simple_rnn_cell_20_kernel_read_readvariableopLsavev2_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_read_readvariableop@savev2_simple_rnn_10_simple_rnn_cell_20_bias_read_readvariableopBsavev2_simple_rnn_11_simple_rnn_cell_21_kernel_read_readvariableopLsavev2_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_read_readvariableop@savev2_simple_rnn_11_simple_rnn_cell_21_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableopIsavev2_adam_simple_rnn_10_simple_rnn_cell_20_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_10_simple_rnn_cell_20_bias_m_read_readvariableopIsavev2_adam_simple_rnn_11_simple_rnn_cell_21_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_11_simple_rnn_cell_21_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableopIsavev2_adam_simple_rnn_10_simple_rnn_cell_20_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_10_simple_rnn_cell_20_bias_v_read_readvariableopIsavev2_adam_simple_rnn_11_simple_rnn_cell_21_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_11_simple_rnn_cell_21_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :2H:H: : : : : :	?:
??:?:	?2:22:2: : : : : : :2H:H:	?:
??:?:	?2:22:2:2H:H:	?:
??:?:	?2:22:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2H: 

_output_shapes
:H:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?2:$ 

_output_shapes

:22: 

_output_shapes
:2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2H: 

_output_shapes
:H:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2H: 

_output_shapes
:H:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:! 

_output_shapes	
:?:%!!

_output_shapes
:	?2:$" 

_output_shapes

:22: #

_output_shapes
:2:$

_output_shapes
: 
??
?
I__inference_sequential_30_layer_call_and_return_conditional_losses_242812

inputsC
?simple_rnn_10_simple_rnn_cell_20_matmul_readvariableop_resourceD
@simple_rnn_10_simple_rnn_cell_20_biasadd_readvariableop_resourceE
Asimple_rnn_10_simple_rnn_cell_20_matmul_1_readvariableop_resourceC
?simple_rnn_11_simple_rnn_cell_21_matmul_readvariableop_resourceD
@simple_rnn_11_simple_rnn_cell_21_biasadd_readvariableop_resourceE
Asimple_rnn_11_simple_rnn_cell_21_matmul_1_readvariableop_resource.
*dense_35_mlcmatmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??dense_35/BiasAdd/ReadVariableOp?!dense_35/MLCMatMul/ReadVariableOp?7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp?6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp?8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp?simple_rnn_10/while?7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp?6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp?8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp?simple_rnn_11/while`
simple_rnn_10/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_10/Shape?
!simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!simple_rnn_10/strided_slice/stack?
#simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_10/strided_slice/stack_1?
#simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_10/strided_slice/stack_2?
simple_rnn_10/strided_sliceStridedSlicesimple_rnn_10/Shape:output:0*simple_rnn_10/strided_slice/stack:output:0,simple_rnn_10/strided_slice/stack_1:output:0,simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_10/strided_slicey
simple_rnn_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_10/zeros/mul/y?
simple_rnn_10/zeros/mulMul$simple_rnn_10/strided_slice:output:0"simple_rnn_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_10/zeros/mul{
simple_rnn_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_10/zeros/Less/y?
simple_rnn_10/zeros/LessLesssimple_rnn_10/zeros/mul:z:0#simple_rnn_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_10/zeros/Less
simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_10/zeros/packed/1?
simple_rnn_10/zeros/packedPack$simple_rnn_10/strided_slice:output:0%simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_10/zeros/packed{
simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_10/zeros/Const?
simple_rnn_10/zerosFill#simple_rnn_10/zeros/packed:output:0"simple_rnn_10/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_10/zeros?
simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_10/transpose/perm?
simple_rnn_10/transpose	Transposeinputs%simple_rnn_10/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
simple_rnn_10/transposey
simple_rnn_10/Shape_1Shapesimple_rnn_10/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_10/Shape_1?
#simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_10/strided_slice_1/stack?
%simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_1/stack_1?
%simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_1/stack_2?
simple_rnn_10/strided_slice_1StridedSlicesimple_rnn_10/Shape_1:output:0,simple_rnn_10/strided_slice_1/stack:output:0.simple_rnn_10/strided_slice_1/stack_1:output:0.simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_10/strided_slice_1?
)simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)simple_rnn_10/TensorArrayV2/element_shape?
simple_rnn_10/TensorArrayV2TensorListReserve2simple_rnn_10/TensorArrayV2/element_shape:output:0&simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_10/TensorArrayV2?
Csimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2E
Csimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape?
5simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_10/transpose:y:0Lsimple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5simple_rnn_10/TensorArrayUnstack/TensorListFromTensor?
#simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_10/strided_slice_2/stack?
%simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_2/stack_1?
%simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_2/stack_2?
simple_rnn_10/strided_slice_2StridedSlicesimple_rnn_10/transpose:y:0,simple_rnn_10/strided_slice_2/stack:output:0.simple_rnn_10/strided_slice_2/stack_1:output:0.simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_10/strided_slice_2?
6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp?simple_rnn_10_simple_rnn_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype028
6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp?
'simple_rnn_10/simple_rnn_cell_20/MatMul	MLCMatMul&simple_rnn_10/strided_slice_2:output:0>simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_10/simple_rnn_cell_20/MatMul?
7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_10_simple_rnn_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
(simple_rnn_10/simple_rnn_cell_20/BiasAddBiasAdd1simple_rnn_10/simple_rnn_cell_20/MatMul:product:0?simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_10/simple_rnn_cell_20/BiasAdd?
8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_10_simple_rnn_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
)simple_rnn_10/simple_rnn_cell_20/MatMul_1	MLCMatMulsimple_rnn_10/zeros:output:0@simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_10/simple_rnn_cell_20/MatMul_1?
$simple_rnn_10/simple_rnn_cell_20/addAddV21simple_rnn_10/simple_rnn_cell_20/BiasAdd:output:03simple_rnn_10/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2&
$simple_rnn_10/simple_rnn_cell_20/add?
%simple_rnn_10/simple_rnn_cell_20/TanhTanh(simple_rnn_10/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2'
%simple_rnn_10/simple_rnn_cell_20/Tanh?
+simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2-
+simple_rnn_10/TensorArrayV2_1/element_shape?
simple_rnn_10/TensorArrayV2_1TensorListReserve4simple_rnn_10/TensorArrayV2_1/element_shape:output:0&simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_10/TensorArrayV2_1j
simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_10/time?
&simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&simple_rnn_10/while/maximum_iterations?
 simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 simple_rnn_10/while/loop_counter?
simple_rnn_10/whileWhile)simple_rnn_10/while/loop_counter:output:0/simple_rnn_10/while/maximum_iterations:output:0simple_rnn_10/time:output:0&simple_rnn_10/TensorArrayV2_1:handle:0simple_rnn_10/zeros:output:0&simple_rnn_10/strided_slice_1:output:0Esimple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_10_simple_rnn_cell_20_matmul_readvariableop_resource@simple_rnn_10_simple_rnn_cell_20_biasadd_readvariableop_resourceAsimple_rnn_10_simple_rnn_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*+
body#R!
simple_rnn_10_while_body_242619*+
cond#R!
simple_rnn_10_while_cond_242618*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_10/while?
>simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape?
0simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_10/while:output:3Gsimple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype022
0simple_rnn_10/TensorArrayV2Stack/TensorListStack?
#simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#simple_rnn_10/strided_slice_3/stack?
%simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%simple_rnn_10/strided_slice_3/stack_1?
%simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_10/strided_slice_3/stack_2?
simple_rnn_10/strided_slice_3StridedSlice9simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_10/strided_slice_3/stack:output:0.simple_rnn_10/strided_slice_3/stack_1:output:0.simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_10/strided_slice_3?
simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
simple_rnn_10/transpose_1/perm?
simple_rnn_10/transpose_1	Transpose9simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_10/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
simple_rnn_10/transpose_1?
activation_25/ReluRelusimple_rnn_10/transpose_1:y:0*
T0*-
_output_shapes
:???????????2
activation_25/Reluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_10/dropout/Const?
dropout_10/dropout/MulMul activation_25/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*-
_output_shapes
:???????????2
dropout_10/dropout/Mulw
dropout_10/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_10/dropout/ratet
dropout_10/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_10/dropout/seed?
dropout_10/dropout
MLCDropout activation_25/Relu:activations:0 dropout_10/dropout/rate:output:0 dropout_10/dropout/seed:output:0*
T0*-
_output_shapes
:???????????2
dropout_10/dropoutu
simple_rnn_11/ShapeShapedropout_10/dropout:output:0*
T0*
_output_shapes
:2
simple_rnn_11/Shape?
!simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!simple_rnn_11/strided_slice/stack?
#simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_11/strided_slice/stack_1?
#simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#simple_rnn_11/strided_slice/stack_2?
simple_rnn_11/strided_sliceStridedSlicesimple_rnn_11/Shape:output:0*simple_rnn_11/strided_slice/stack:output:0,simple_rnn_11/strided_slice/stack_1:output:0,simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_11/strided_slicex
simple_rnn_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_11/zeros/mul/y?
simple_rnn_11/zeros/mulMul$simple_rnn_11/strided_slice:output:0"simple_rnn_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_11/zeros/mul{
simple_rnn_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_11/zeros/Less/y?
simple_rnn_11/zeros/LessLesssimple_rnn_11/zeros/mul:z:0#simple_rnn_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_11/zeros/Less~
simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_11/zeros/packed/1?
simple_rnn_11/zeros/packedPack$simple_rnn_11/strided_slice:output:0%simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_11/zeros/packed{
simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_11/zeros/Const?
simple_rnn_11/zerosFill#simple_rnn_11/zeros/packed:output:0"simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
simple_rnn_11/zeros?
simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_11/transpose/perm?
simple_rnn_11/transpose	Transposedropout_10/dropout:output:0%simple_rnn_11/transpose/perm:output:0*
T0*-
_output_shapes
:???????????2
simple_rnn_11/transposey
simple_rnn_11/Shape_1Shapesimple_rnn_11/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_11/Shape_1?
#simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_11/strided_slice_1/stack?
%simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_1/stack_1?
%simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_1/stack_2?
simple_rnn_11/strided_slice_1StridedSlicesimple_rnn_11/Shape_1:output:0,simple_rnn_11/strided_slice_1/stack:output:0.simple_rnn_11/strided_slice_1/stack_1:output:0.simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_11/strided_slice_1?
)simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)simple_rnn_11/TensorArrayV2/element_shape?
simple_rnn_11/TensorArrayV2TensorListReserve2simple_rnn_11/TensorArrayV2/element_shape:output:0&simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_11/TensorArrayV2?
Csimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2E
Csimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape?
5simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_11/transpose:y:0Lsimple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5simple_rnn_11/TensorArrayUnstack/TensorListFromTensor?
#simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#simple_rnn_11/strided_slice_2/stack?
%simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_2/stack_1?
%simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_2/stack_2?
simple_rnn_11/strided_slice_2StridedSlicesimple_rnn_11/transpose:y:0,simple_rnn_11/strided_slice_2/stack:output:0.simple_rnn_11/strided_slice_2/stack_1:output:0.simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_11/strided_slice_2?
6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOp?simple_rnn_11_simple_rnn_cell_21_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype028
6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp?
'simple_rnn_11/simple_rnn_cell_21/MatMul	MLCMatMul&simple_rnn_11/strided_slice_2:output:0>simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22)
'simple_rnn_11/simple_rnn_cell_21/MatMul?
7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_11_simple_rnn_cell_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype029
7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
(simple_rnn_11/simple_rnn_cell_21/BiasAddBiasAdd1simple_rnn_11/simple_rnn_cell_21/MatMul:product:0?simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22*
(simple_rnn_11/simple_rnn_cell_21/BiasAdd?
8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_11_simple_rnn_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02:
8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
)simple_rnn_11/simple_rnn_cell_21/MatMul_1	MLCMatMulsimple_rnn_11/zeros:output:0@simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22+
)simple_rnn_11/simple_rnn_cell_21/MatMul_1?
$simple_rnn_11/simple_rnn_cell_21/addAddV21simple_rnn_11/simple_rnn_cell_21/BiasAdd:output:03simple_rnn_11/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22&
$simple_rnn_11/simple_rnn_cell_21/add?
%simple_rnn_11/simple_rnn_cell_21/TanhTanh(simple_rnn_11/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????22'
%simple_rnn_11/simple_rnn_cell_21/Tanh?
+simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2-
+simple_rnn_11/TensorArrayV2_1/element_shape?
simple_rnn_11/TensorArrayV2_1TensorListReserve4simple_rnn_11/TensorArrayV2_1/element_shape:output:0&simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_11/TensorArrayV2_1j
simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_11/time?
&simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&simple_rnn_11/while/maximum_iterations?
 simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 simple_rnn_11/while/loop_counter?
simple_rnn_11/whileWhile)simple_rnn_11/while/loop_counter:output:0/simple_rnn_11/while/maximum_iterations:output:0simple_rnn_11/time:output:0&simple_rnn_11/TensorArrayV2_1:handle:0simple_rnn_11/zeros:output:0&simple_rnn_11/strided_slice_1:output:0Esimple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_11_simple_rnn_cell_21_matmul_readvariableop_resource@simple_rnn_11_simple_rnn_cell_21_biasadd_readvariableop_resourceAsimple_rnn_11_simple_rnn_cell_21_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*+
body#R!
simple_rnn_11_while_body_242733*+
cond#R!
simple_rnn_11_while_cond_242732*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
simple_rnn_11/while?
>simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape?
0simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_11/while:output:3Gsimple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype022
0simple_rnn_11/TensorArrayV2Stack/TensorListStack?
#simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#simple_rnn_11/strided_slice_3/stack?
%simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%simple_rnn_11/strided_slice_3/stack_1?
%simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%simple_rnn_11/strided_slice_3/stack_2?
simple_rnn_11/strided_slice_3StridedSlice9simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_11/strided_slice_3/stack:output:0.simple_rnn_11/strided_slice_3/stack_1:output:0.simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
simple_rnn_11/strided_slice_3?
simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
simple_rnn_11/transpose_1/perm?
simple_rnn_11/transpose_1	Transpose9simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_11/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22
simple_rnn_11/transpose_1?
activation_26/ReluRelu&simple_rnn_11/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22
activation_26/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_11/dropout/Const?
dropout_11/dropout/MulMul activation_26/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_11/dropout/Mulw
dropout_11/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_11/dropout/ratet
dropout_11/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_11/dropout/seed?
dropout_11/dropout
MLCDropout activation_26/Relu:activations:0 dropout_11/dropout/rate:output:0 dropout_11/dropout/seed:output:0*
T0*'
_output_shapes
:?????????22
dropout_11/dropout?
!dense_35/MLCMatMul/ReadVariableOpReadVariableOp*dense_35_mlcmatmul_readvariableop_resource*
_output_shapes

:2H*
dtype02#
!dense_35/MLCMatMul/ReadVariableOp?
dense_35/MLCMatMul	MLCMatMuldropout_11/dropout:output:0)dense_35/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_35/MLCMatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MLCMatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_35/BiasAdds
dense_35/TanhTanhdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_35/Tanh?
IdentityIdentitydense_35/Tanh:y:0 ^dense_35/BiasAdd/ReadVariableOp"^dense_35/MLCMatMul/ReadVariableOp8^simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp7^simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp9^simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp^simple_rnn_10/while8^simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp7^simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp9^simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp^simple_rnn_11/while*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2F
!dense_35/MLCMatMul/ReadVariableOp!dense_35/MLCMatMul/ReadVariableOp2r
7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp7simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp2p
6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp6simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp2t
8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp8simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp2*
simple_rnn_10/whilesimple_rnn_10/while2r
7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp7simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp2p
6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp6simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp2t
8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp8simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp2*
simple_rnn_11/whilesimple_rnn_11/while:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?4
?
while_body_243377
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_simple_rnn_cell_20_matmul_readvariableop_resource_0>
:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0?
;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_simple_rnn_cell_20_matmul_readvariableop_resource<
8while_simple_rnn_cell_20_biasadd_readvariableop_resource=
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource??/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?.while/simple_rnn_cell_20/MatMul/ReadVariableOp?0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp9while_simple_rnn_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.while/simple_rnn_cell_20/MatMul/ReadVariableOp?
while/simple_rnn_cell_20/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_20/MatMul?
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
 while/simple_rnn_cell_20/BiasAddBiasAdd)while/simple_rnn_cell_20/MatMul:product:07while/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_20/BiasAdd?
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
!while/simple_rnn_cell_20/MatMul_1	MLCMatMulwhile_placeholder_28while/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/simple_rnn_cell_20/MatMul_1?
while/simple_rnn_cell_20/addAddV2)while/simple_rnn_cell_20/BiasAdd:output:0+while/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/add?
while/simple_rnn_cell_20/TanhTanh while/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_20/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder!while/simple_rnn_cell_20/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity!while/simple_rnn_cell_20/Tanh:y:00^while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/^while/simple_rnn_cell_20/MatMul/ReadVariableOp1^while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"v
8while_simple_rnn_cell_20_biasadd_readvariableop_resource:while_simple_rnn_cell_20_biasadd_readvariableop_resource_0"x
9while_simple_rnn_cell_20_matmul_1_readvariableop_resource;while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0"t
7while_simple_rnn_cell_20_matmul_readvariableop_resource9while_simple_rnn_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp2`
.while/simple_rnn_cell_20/MatMul/ReadVariableOp.while/simple_rnn_cell_20/MatMul/ReadVariableOp2d
0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp0while/simple_rnn_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_dropout_10_layer_call_fn_243606

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2420792
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?D
?
simple_rnn_10_while_body_2428588
4simple_rnn_10_while_simple_rnn_10_while_loop_counter>
:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations#
simple_rnn_10_while_placeholder%
!simple_rnn_10_while_placeholder_1%
!simple_rnn_10_while_placeholder_27
3simple_rnn_10_while_simple_rnn_10_strided_slice_1_0s
osimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0K
Gsimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resource_0L
Hsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resource_0M
Isimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0 
simple_rnn_10_while_identity"
simple_rnn_10_while_identity_1"
simple_rnn_10_while_identity_2"
simple_rnn_10_while_identity_3"
simple_rnn_10_while_identity_45
1simple_rnn_10_while_simple_rnn_10_strided_slice_1q
msimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorI
Esimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resourceJ
Fsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resourceK
Gsimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource??=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
Esimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Esimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape?
7simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_10_while_placeholderNsimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype029
7simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem?
<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02>
<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?
-simple_rnn_10/while/simple_rnn_cell_20/MatMul	MLCMatMul>simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_10/while/simple_rnn_cell_20/MatMul?
=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02?
=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
.simple_rnn_10/while/simple_rnn_cell_20/BiasAddBiasAdd7simple_rnn_10/while/simple_rnn_cell_20/MatMul:product:0Esimple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.simple_rnn_10/while/simple_rnn_cell_20/BiasAdd?
>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02@
>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1	MLCMatMul!simple_rnn_10_while_placeholder_2Fsimple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1?
*simple_rnn_10/while/simple_rnn_cell_20/addAddV27simple_rnn_10/while/simple_rnn_cell_20/BiasAdd:output:09simple_rnn_10/while/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2,
*simple_rnn_10/while/simple_rnn_cell_20/add?
+simple_rnn_10/while/simple_rnn_cell_20/TanhTanh.simple_rnn_10/while/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2-
+simple_rnn_10/while/simple_rnn_cell_20/Tanh?
8simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_10_while_placeholder_1simple_rnn_10_while_placeholder/simple_rnn_10/while/simple_rnn_cell_20/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemx
simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_10/while/add/y?
simple_rnn_10/while/addAddV2simple_rnn_10_while_placeholder"simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_10/while/add|
simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_10/while/add_1/y?
simple_rnn_10/while/add_1AddV24simple_rnn_10_while_simple_rnn_10_while_loop_counter$simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_10/while/add_1?
simple_rnn_10/while/IdentityIdentitysimple_rnn_10/while/add_1:z:0>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_10/while/Identity?
simple_rnn_10/while/Identity_1Identity:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_10/while/Identity_1?
simple_rnn_10/while/Identity_2Identitysimple_rnn_10/while/add:z:0>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_10/while/Identity_2?
simple_rnn_10/while/Identity_3IdentityHsimple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_10/while/Identity_3?
simple_rnn_10/while/Identity_4Identity/simple_rnn_10/while/simple_rnn_cell_20/Tanh:y:0>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2 
simple_rnn_10/while/Identity_4"E
simple_rnn_10_while_identity%simple_rnn_10/while/Identity:output:0"I
simple_rnn_10_while_identity_1'simple_rnn_10/while/Identity_1:output:0"I
simple_rnn_10_while_identity_2'simple_rnn_10/while/Identity_2:output:0"I
simple_rnn_10_while_identity_3'simple_rnn_10/while/Identity_3:output:0"I
simple_rnn_10_while_identity_4'simple_rnn_10/while/Identity_4:output:0"h
1simple_rnn_10_while_simple_rnn_10_strided_slice_13simple_rnn_10_while_simple_rnn_10_strided_slice_1_0"?
Fsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resourceHsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resource_0"?
Gsimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resourceIsimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0"?
Esimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resourceGsimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resource_0"?
msimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorosimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2~
=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp2|
<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp2?
>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?D
?
simple_rnn_10_while_body_2426198
4simple_rnn_10_while_simple_rnn_10_while_loop_counter>
:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations#
simple_rnn_10_while_placeholder%
!simple_rnn_10_while_placeholder_1%
!simple_rnn_10_while_placeholder_27
3simple_rnn_10_while_simple_rnn_10_strided_slice_1_0s
osimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0K
Gsimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resource_0L
Hsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resource_0M
Isimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0 
simple_rnn_10_while_identity"
simple_rnn_10_while_identity_1"
simple_rnn_10_while_identity_2"
simple_rnn_10_while_identity_3"
simple_rnn_10_while_identity_45
1simple_rnn_10_while_simple_rnn_10_strided_slice_1q
msimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorI
Esimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resourceJ
Fsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resourceK
Gsimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource??=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
Esimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Esimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape?
7simple_rnn_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_10_while_placeholderNsimple_rnn_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype029
7simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem?
<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02>
<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?
-simple_rnn_10/while/simple_rnn_cell_20/MatMul	MLCMatMul>simple_rnn_10/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_10/while/simple_rnn_cell_20/MatMul?
=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02?
=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
.simple_rnn_10/while/simple_rnn_cell_20/BiasAddBiasAdd7simple_rnn_10/while/simple_rnn_cell_20/MatMul:product:0Esimple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.simple_rnn_10/while/simple_rnn_cell_20/BiasAdd?
>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02@
>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1	MLCMatMul!simple_rnn_10_while_placeholder_2Fsimple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/simple_rnn_10/while/simple_rnn_cell_20/MatMul_1?
*simple_rnn_10/while/simple_rnn_cell_20/addAddV27simple_rnn_10/while/simple_rnn_cell_20/BiasAdd:output:09simple_rnn_10/while/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2,
*simple_rnn_10/while/simple_rnn_cell_20/add?
+simple_rnn_10/while/simple_rnn_cell_20/TanhTanh.simple_rnn_10/while/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2-
+simple_rnn_10/while/simple_rnn_cell_20/Tanh?
8simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_10_while_placeholder_1simple_rnn_10_while_placeholder/simple_rnn_10/while/simple_rnn_cell_20/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8simple_rnn_10/while/TensorArrayV2Write/TensorListSetItemx
simple_rnn_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_10/while/add/y?
simple_rnn_10/while/addAddV2simple_rnn_10_while_placeholder"simple_rnn_10/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_10/while/add|
simple_rnn_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_10/while/add_1/y?
simple_rnn_10/while/add_1AddV24simple_rnn_10_while_simple_rnn_10_while_loop_counter$simple_rnn_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_10/while/add_1?
simple_rnn_10/while/IdentityIdentitysimple_rnn_10/while/add_1:z:0>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_10/while/Identity?
simple_rnn_10/while/Identity_1Identity:simple_rnn_10_while_simple_rnn_10_while_maximum_iterations>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_10/while/Identity_1?
simple_rnn_10/while/Identity_2Identitysimple_rnn_10/while/add:z:0>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_10/while/Identity_2?
simple_rnn_10/while/Identity_3IdentityHsimple_rnn_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
simple_rnn_10/while/Identity_3?
simple_rnn_10/while/Identity_4Identity/simple_rnn_10/while/simple_rnn_cell_20/Tanh:y:0>^simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=^simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp?^simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2 
simple_rnn_10/while/Identity_4"E
simple_rnn_10_while_identity%simple_rnn_10/while/Identity:output:0"I
simple_rnn_10_while_identity_1'simple_rnn_10/while/Identity_1:output:0"I
simple_rnn_10_while_identity_2'simple_rnn_10/while/Identity_2:output:0"I
simple_rnn_10_while_identity_3'simple_rnn_10/while/Identity_3:output:0"I
simple_rnn_10_while_identity_4'simple_rnn_10/while/Identity_4:output:0"h
1simple_rnn_10_while_simple_rnn_10_strided_slice_13simple_rnn_10_while_simple_rnn_10_strided_slice_1_0"?
Fsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resourceHsimple_rnn_10_while_simple_rnn_cell_20_biasadd_readvariableop_resource_0"?
Gsimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resourceIsimple_rnn_10_while_simple_rnn_cell_20_matmul_1_readvariableop_resource_0"?
Esimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resourceGsimple_rnn_10_while_simple_rnn_cell_20_matmul_readvariableop_resource_0"?
msimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensorosimple_rnn_10_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_10_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2~
=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp=simple_rnn_10/while/simple_rnn_cell_20/BiasAdd/ReadVariableOp2|
<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp<simple_rnn_10/while/simple_rnn_cell_20/MatMul/ReadVariableOp2?
>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp>simple_rnn_10/while/simple_rnn_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_sequential_30_layer_call_fn_243085

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_30_layer_call_and_return_conditional_losses_2425232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_dropout_11_layer_call_and_return_conditional_losses_242372

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*'
_output_shapes
:?????????22	
dropoutd
IdentityIdentitydropout:output:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
G
+__inference_dropout_10_layer_call_fn_243611

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2420842
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
while_cond_241215
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241215___redundant_placeholder04
0while_while_cond_241215___redundant_placeholder14
0while_while_cond_241215___redundant_placeholder24
0while_while_cond_241215___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_243768
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_243768___redundant_placeholder04
0while_while_cond_243768___redundant_placeholder14
0while_while_cond_243768___redundant_placeholder24
0while_while_cond_243768___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?	
?
3__inference_simple_rnn_cell_21_layer_call_fn_244267

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_2413372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?
?
while_cond_241960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241960___redundant_placeholder04
0while_while_cond_241960___redundant_placeholder14
0while_while_cond_241960___redundant_placeholder24
0while_while_cond_241960___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
e
I__inference_activation_25_layer_call_and_return_conditional_losses_242062

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_244524
file_prefix$
 assignvariableop_dense_35_kernel$
 assignvariableop_1_dense_35_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate>
:assignvariableop_7_simple_rnn_10_simple_rnn_cell_20_kernelH
Dassignvariableop_8_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel<
8assignvariableop_9_simple_rnn_10_simple_rnn_cell_20_bias?
;assignvariableop_10_simple_rnn_11_simple_rnn_cell_21_kernelI
Eassignvariableop_11_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel=
9assignvariableop_12_simple_rnn_11_simple_rnn_cell_21_bias
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1
assignvariableop_17_total_2
assignvariableop_18_count_2.
*assignvariableop_19_adam_dense_35_kernel_m,
(assignvariableop_20_adam_dense_35_bias_mF
Bassignvariableop_21_adam_simple_rnn_10_simple_rnn_cell_20_kernel_mP
Lassignvariableop_22_adam_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_mD
@assignvariableop_23_adam_simple_rnn_10_simple_rnn_cell_20_bias_mF
Bassignvariableop_24_adam_simple_rnn_11_simple_rnn_cell_21_kernel_mP
Lassignvariableop_25_adam_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_mD
@assignvariableop_26_adam_simple_rnn_11_simple_rnn_cell_21_bias_m.
*assignvariableop_27_adam_dense_35_kernel_v,
(assignvariableop_28_adam_dense_35_bias_vF
Bassignvariableop_29_adam_simple_rnn_10_simple_rnn_cell_20_kernel_vP
Lassignvariableop_30_adam_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_vD
@assignvariableop_31_adam_simple_rnn_10_simple_rnn_cell_20_bias_vF
Bassignvariableop_32_adam_simple_rnn_11_simple_rnn_cell_21_kernel_vP
Lassignvariableop_33_adam_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_vD
@assignvariableop_34_adam_simple_rnn_11_simple_rnn_cell_21_bias_v
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_35_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_35_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp:assignvariableop_7_simple_rnn_10_simple_rnn_cell_20_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpDassignvariableop_8_simple_rnn_10_simple_rnn_cell_20_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp8assignvariableop_9_simple_rnn_10_simple_rnn_cell_20_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp;assignvariableop_10_simple_rnn_11_simple_rnn_cell_21_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpEassignvariableop_11_simple_rnn_11_simple_rnn_cell_21_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_simple_rnn_11_simple_rnn_cell_21_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_35_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_35_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpBassignvariableop_21_adam_simple_rnn_10_simple_rnn_cell_20_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpLassignvariableop_22_adam_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adam_simple_rnn_10_simple_rnn_cell_20_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpBassignvariableop_24_adam_simple_rnn_11_simple_rnn_cell_21_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpLassignvariableop_25_adam_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_simple_rnn_11_simple_rnn_cell_21_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_35_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_35_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpBassignvariableop_29_adam_simple_rnn_10_simple_rnn_cell_20_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpLassignvariableop_30_adam_simple_rnn_10_simple_rnn_cell_20_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_simple_rnn_10_simple_rnn_cell_20_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpBassignvariableop_32_adam_simple_rnn_11_simple_rnn_cell_21_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpLassignvariableop_33_adam_simple_rnn_11_simple_rnn_cell_21_recurrent_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp@assignvariableop_34_adam_simple_rnn_11_simple_rnn_cell_21_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35?
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
I__inference_sequential_30_layer_call_and_return_conditional_losses_242523

inputs
simple_rnn_10_242499
simple_rnn_10_242501
simple_rnn_10_242503
simple_rnn_11_242508
simple_rnn_11_242510
simple_rnn_11_242512
dense_35_242517
dense_35_242519
identity?? dense_35/StatefulPartitionedCall?%simple_rnn_10/StatefulPartitionedCall?%simple_rnn_11/StatefulPartitionedCall?
%simple_rnn_10/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_10_242499simple_rnn_10_242501simple_rnn_10_242503*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_2420272'
%simple_rnn_10/StatefulPartitionedCall?
activation_25/PartitionedCallPartitionedCall.simple_rnn_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_25_layer_call_and_return_conditional_losses_2420622
activation_25/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall&activation_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_2420842
dropout_10/PartitionedCall?
%simple_rnn_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0simple_rnn_11_242508simple_rnn_11_242510simple_rnn_11_242512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_2423202'
%simple_rnn_11/StatefulPartitionedCall?
activation_26/PartitionedCallPartitionedCall.simple_rnn_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_26_layer_call_and_return_conditional_losses_2423552
activation_26/PartitionedCall?
dropout_11/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_2423772
dropout_11/PartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_35_242517dense_35_242519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_2424012"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_35/StatefulPartitionedCall&^simple_rnn_10/StatefulPartitionedCall&^simple_rnn_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%simple_rnn_10/StatefulPartitionedCall%simple_rnn_10/StatefulPartitionedCall2N
%simple_rnn_11/StatefulPartitionedCall%simple_rnn_11/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
3__inference_simple_rnn_cell_20_layer_call_fn_244219

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_2408422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:?????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
e
I__inference_activation_26_layer_call_and_return_conditional_losses_244108

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????2:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?

!__inference__wrapped_model_240776
simple_rnn_10_inputQ
Msequential_30_simple_rnn_10_simple_rnn_cell_20_matmul_readvariableop_resourceR
Nsequential_30_simple_rnn_10_simple_rnn_cell_20_biasadd_readvariableop_resourceS
Osequential_30_simple_rnn_10_simple_rnn_cell_20_matmul_1_readvariableop_resourceQ
Msequential_30_simple_rnn_11_simple_rnn_cell_21_matmul_readvariableop_resourceR
Nsequential_30_simple_rnn_11_simple_rnn_cell_21_biasadd_readvariableop_resourceS
Osequential_30_simple_rnn_11_simple_rnn_cell_21_matmul_1_readvariableop_resource<
8sequential_30_dense_35_mlcmatmul_readvariableop_resource:
6sequential_30_dense_35_biasadd_readvariableop_resource
identity??-sequential_30/dense_35/BiasAdd/ReadVariableOp?/sequential_30/dense_35/MLCMatMul/ReadVariableOp?Esequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp?Dsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp?Fsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp?!sequential_30/simple_rnn_10/while?Esequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp?Dsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp?Fsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp?!sequential_30/simple_rnn_11/while?
!sequential_30/simple_rnn_10/ShapeShapesimple_rnn_10_input*
T0*
_output_shapes
:2#
!sequential_30/simple_rnn_10/Shape?
/sequential_30/simple_rnn_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_30/simple_rnn_10/strided_slice/stack?
1sequential_30/simple_rnn_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_30/simple_rnn_10/strided_slice/stack_1?
1sequential_30/simple_rnn_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_30/simple_rnn_10/strided_slice/stack_2?
)sequential_30/simple_rnn_10/strided_sliceStridedSlice*sequential_30/simple_rnn_10/Shape:output:08sequential_30/simple_rnn_10/strided_slice/stack:output:0:sequential_30/simple_rnn_10/strided_slice/stack_1:output:0:sequential_30/simple_rnn_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_30/simple_rnn_10/strided_slice?
'sequential_30/simple_rnn_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2)
'sequential_30/simple_rnn_10/zeros/mul/y?
%sequential_30/simple_rnn_10/zeros/mulMul2sequential_30/simple_rnn_10/strided_slice:output:00sequential_30/simple_rnn_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2'
%sequential_30/simple_rnn_10/zeros/mul?
(sequential_30/simple_rnn_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_30/simple_rnn_10/zeros/Less/y?
&sequential_30/simple_rnn_10/zeros/LessLess)sequential_30/simple_rnn_10/zeros/mul:z:01sequential_30/simple_rnn_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2(
&sequential_30/simple_rnn_10/zeros/Less?
*sequential_30/simple_rnn_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2,
*sequential_30/simple_rnn_10/zeros/packed/1?
(sequential_30/simple_rnn_10/zeros/packedPack2sequential_30/simple_rnn_10/strided_slice:output:03sequential_30/simple_rnn_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(sequential_30/simple_rnn_10/zeros/packed?
'sequential_30/simple_rnn_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'sequential_30/simple_rnn_10/zeros/Const?
!sequential_30/simple_rnn_10/zerosFill1sequential_30/simple_rnn_10/zeros/packed:output:00sequential_30/simple_rnn_10/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2#
!sequential_30/simple_rnn_10/zeros?
*sequential_30/simple_rnn_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_30/simple_rnn_10/transpose/perm?
%sequential_30/simple_rnn_10/transpose	Transposesimple_rnn_10_input3sequential_30/simple_rnn_10/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2'
%sequential_30/simple_rnn_10/transpose?
#sequential_30/simple_rnn_10/Shape_1Shape)sequential_30/simple_rnn_10/transpose:y:0*
T0*
_output_shapes
:2%
#sequential_30/simple_rnn_10/Shape_1?
1sequential_30/simple_rnn_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_30/simple_rnn_10/strided_slice_1/stack?
3sequential_30/simple_rnn_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_10/strided_slice_1/stack_1?
3sequential_30/simple_rnn_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_10/strided_slice_1/stack_2?
+sequential_30/simple_rnn_10/strided_slice_1StridedSlice,sequential_30/simple_rnn_10/Shape_1:output:0:sequential_30/simple_rnn_10/strided_slice_1/stack:output:0<sequential_30/simple_rnn_10/strided_slice_1/stack_1:output:0<sequential_30/simple_rnn_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_30/simple_rnn_10/strided_slice_1?
7sequential_30/simple_rnn_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7sequential_30/simple_rnn_10/TensorArrayV2/element_shape?
)sequential_30/simple_rnn_10/TensorArrayV2TensorListReserve@sequential_30/simple_rnn_10/TensorArrayV2/element_shape:output:04sequential_30/simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_30/simple_rnn_10/TensorArrayV2?
Qsequential_30/simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2S
Qsequential_30/simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape?
Csequential_30/simple_rnn_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_30/simple_rnn_10/transpose:y:0Zsequential_30/simple_rnn_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02E
Csequential_30/simple_rnn_10/TensorArrayUnstack/TensorListFromTensor?
1sequential_30/simple_rnn_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_30/simple_rnn_10/strided_slice_2/stack?
3sequential_30/simple_rnn_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_10/strided_slice_2/stack_1?
3sequential_30/simple_rnn_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_10/strided_slice_2/stack_2?
+sequential_30/simple_rnn_10/strided_slice_2StridedSlice)sequential_30/simple_rnn_10/transpose:y:0:sequential_30/simple_rnn_10/strided_slice_2/stack:output:0<sequential_30/simple_rnn_10/strided_slice_2/stack_1:output:0<sequential_30/simple_rnn_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2-
+sequential_30/simple_rnn_10/strided_slice_2?
Dsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOpMsequential_30_simple_rnn_10_simple_rnn_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
Dsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp?
5sequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul	MLCMatMul4sequential_30/simple_rnn_10/strided_slice_2:output:0Lsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5sequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul?
Esequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOpNsequential_30_simple_rnn_10_simple_rnn_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
Esequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp?
6sequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAddBiasAdd?sequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul:product:0Msequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????28
6sequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAdd?
Fsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOpOsequential_30_simple_rnn_10_simple_rnn_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02H
Fsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp?
7sequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1	MLCMatMul*sequential_30/simple_rnn_10/zeros:output:0Nsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7sequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1?
2sequential_30/simple_rnn_10/simple_rnn_cell_20/addAddV2?sequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAdd:output:0Asequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????24
2sequential_30/simple_rnn_10/simple_rnn_cell_20/add?
3sequential_30/simple_rnn_10/simple_rnn_cell_20/TanhTanh6sequential_30/simple_rnn_10/simple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????25
3sequential_30/simple_rnn_10/simple_rnn_cell_20/Tanh?
9sequential_30/simple_rnn_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2;
9sequential_30/simple_rnn_10/TensorArrayV2_1/element_shape?
+sequential_30/simple_rnn_10/TensorArrayV2_1TensorListReserveBsequential_30/simple_rnn_10/TensorArrayV2_1/element_shape:output:04sequential_30/simple_rnn_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+sequential_30/simple_rnn_10/TensorArrayV2_1?
 sequential_30/simple_rnn_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_30/simple_rnn_10/time?
4sequential_30/simple_rnn_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4sequential_30/simple_rnn_10/while/maximum_iterations?
.sequential_30/simple_rnn_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_30/simple_rnn_10/while/loop_counter?
!sequential_30/simple_rnn_10/whileWhile7sequential_30/simple_rnn_10/while/loop_counter:output:0=sequential_30/simple_rnn_10/while/maximum_iterations:output:0)sequential_30/simple_rnn_10/time:output:04sequential_30/simple_rnn_10/TensorArrayV2_1:handle:0*sequential_30/simple_rnn_10/zeros:output:04sequential_30/simple_rnn_10/strided_slice_1:output:0Ssequential_30/simple_rnn_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_30_simple_rnn_10_simple_rnn_cell_20_matmul_readvariableop_resourceNsequential_30_simple_rnn_10_simple_rnn_cell_20_biasadd_readvariableop_resourceOsequential_30_simple_rnn_10_simple_rnn_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*9
body1R/
-sequential_30_simple_rnn_10_while_body_240591*9
cond1R/
-sequential_30_simple_rnn_10_while_cond_240590*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2#
!sequential_30/simple_rnn_10/while?
Lsequential_30/simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2N
Lsequential_30/simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape?
>sequential_30/simple_rnn_10/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_30/simple_rnn_10/while:output:3Usequential_30/simple_rnn_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02@
>sequential_30/simple_rnn_10/TensorArrayV2Stack/TensorListStack?
1sequential_30/simple_rnn_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1sequential_30/simple_rnn_10/strided_slice_3/stack?
3sequential_30/simple_rnn_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_30/simple_rnn_10/strided_slice_3/stack_1?
3sequential_30/simple_rnn_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_10/strided_slice_3/stack_2?
+sequential_30/simple_rnn_10/strided_slice_3StridedSliceGsequential_30/simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_30/simple_rnn_10/strided_slice_3/stack:output:0<sequential_30/simple_rnn_10/strided_slice_3/stack_1:output:0<sequential_30/simple_rnn_10/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2-
+sequential_30/simple_rnn_10/strided_slice_3?
,sequential_30/simple_rnn_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,sequential_30/simple_rnn_10/transpose_1/perm?
'sequential_30/simple_rnn_10/transpose_1	TransposeGsequential_30/simple_rnn_10/TensorArrayV2Stack/TensorListStack:tensor:05sequential_30/simple_rnn_10/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2)
'sequential_30/simple_rnn_10/transpose_1?
 sequential_30/activation_25/ReluRelu+sequential_30/simple_rnn_10/transpose_1:y:0*
T0*-
_output_shapes
:???????????2"
 sequential_30/activation_25/Relu?
!sequential_30/dropout_10/IdentityIdentity.sequential_30/activation_25/Relu:activations:0*
T0*-
_output_shapes
:???????????2#
!sequential_30/dropout_10/Identity?
!sequential_30/simple_rnn_11/ShapeShape*sequential_30/dropout_10/Identity:output:0*
T0*
_output_shapes
:2#
!sequential_30/simple_rnn_11/Shape?
/sequential_30/simple_rnn_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_30/simple_rnn_11/strided_slice/stack?
1sequential_30/simple_rnn_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_30/simple_rnn_11/strided_slice/stack_1?
1sequential_30/simple_rnn_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_30/simple_rnn_11/strided_slice/stack_2?
)sequential_30/simple_rnn_11/strided_sliceStridedSlice*sequential_30/simple_rnn_11/Shape:output:08sequential_30/simple_rnn_11/strided_slice/stack:output:0:sequential_30/simple_rnn_11/strided_slice/stack_1:output:0:sequential_30/simple_rnn_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_30/simple_rnn_11/strided_slice?
'sequential_30/simple_rnn_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22)
'sequential_30/simple_rnn_11/zeros/mul/y?
%sequential_30/simple_rnn_11/zeros/mulMul2sequential_30/simple_rnn_11/strided_slice:output:00sequential_30/simple_rnn_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2'
%sequential_30/simple_rnn_11/zeros/mul?
(sequential_30/simple_rnn_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_30/simple_rnn_11/zeros/Less/y?
&sequential_30/simple_rnn_11/zeros/LessLess)sequential_30/simple_rnn_11/zeros/mul:z:01sequential_30/simple_rnn_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2(
&sequential_30/simple_rnn_11/zeros/Less?
*sequential_30/simple_rnn_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22,
*sequential_30/simple_rnn_11/zeros/packed/1?
(sequential_30/simple_rnn_11/zeros/packedPack2sequential_30/simple_rnn_11/strided_slice:output:03sequential_30/simple_rnn_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(sequential_30/simple_rnn_11/zeros/packed?
'sequential_30/simple_rnn_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'sequential_30/simple_rnn_11/zeros/Const?
!sequential_30/simple_rnn_11/zerosFill1sequential_30/simple_rnn_11/zeros/packed:output:00sequential_30/simple_rnn_11/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22#
!sequential_30/simple_rnn_11/zeros?
*sequential_30/simple_rnn_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_30/simple_rnn_11/transpose/perm?
%sequential_30/simple_rnn_11/transpose	Transpose*sequential_30/dropout_10/Identity:output:03sequential_30/simple_rnn_11/transpose/perm:output:0*
T0*-
_output_shapes
:???????????2'
%sequential_30/simple_rnn_11/transpose?
#sequential_30/simple_rnn_11/Shape_1Shape)sequential_30/simple_rnn_11/transpose:y:0*
T0*
_output_shapes
:2%
#sequential_30/simple_rnn_11/Shape_1?
1sequential_30/simple_rnn_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_30/simple_rnn_11/strided_slice_1/stack?
3sequential_30/simple_rnn_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_11/strided_slice_1/stack_1?
3sequential_30/simple_rnn_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_11/strided_slice_1/stack_2?
+sequential_30/simple_rnn_11/strided_slice_1StridedSlice,sequential_30/simple_rnn_11/Shape_1:output:0:sequential_30/simple_rnn_11/strided_slice_1/stack:output:0<sequential_30/simple_rnn_11/strided_slice_1/stack_1:output:0<sequential_30/simple_rnn_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_30/simple_rnn_11/strided_slice_1?
7sequential_30/simple_rnn_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7sequential_30/simple_rnn_11/TensorArrayV2/element_shape?
)sequential_30/simple_rnn_11/TensorArrayV2TensorListReserve@sequential_30/simple_rnn_11/TensorArrayV2/element_shape:output:04sequential_30/simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_30/simple_rnn_11/TensorArrayV2?
Qsequential_30/simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2S
Qsequential_30/simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape?
Csequential_30/simple_rnn_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_30/simple_rnn_11/transpose:y:0Zsequential_30/simple_rnn_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02E
Csequential_30/simple_rnn_11/TensorArrayUnstack/TensorListFromTensor?
1sequential_30/simple_rnn_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_30/simple_rnn_11/strided_slice_2/stack?
3sequential_30/simple_rnn_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_11/strided_slice_2/stack_1?
3sequential_30/simple_rnn_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_11/strided_slice_2/stack_2?
+sequential_30/simple_rnn_11/strided_slice_2StridedSlice)sequential_30/simple_rnn_11/transpose:y:0:sequential_30/simple_rnn_11/strided_slice_2/stack:output:0<sequential_30/simple_rnn_11/strided_slice_2/stack_1:output:0<sequential_30/simple_rnn_11/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2-
+sequential_30/simple_rnn_11/strided_slice_2?
Dsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOpReadVariableOpMsequential_30_simple_rnn_11_simple_rnn_cell_21_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02F
Dsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp?
5sequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul	MLCMatMul4sequential_30/simple_rnn_11/strided_slice_2:output:0Lsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????227
5sequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul?
Esequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOpReadVariableOpNsequential_30_simple_rnn_11_simple_rnn_cell_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02G
Esequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp?
6sequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAddBiasAdd?sequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul:product:0Msequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????228
6sequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAdd?
Fsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOpReadVariableOpOsequential_30_simple_rnn_11_simple_rnn_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02H
Fsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp?
7sequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1	MLCMatMul*sequential_30/simple_rnn_11/zeros:output:0Nsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????229
7sequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1?
2sequential_30/simple_rnn_11/simple_rnn_cell_21/addAddV2?sequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAdd:output:0Asequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:?????????224
2sequential_30/simple_rnn_11/simple_rnn_cell_21/add?
3sequential_30/simple_rnn_11/simple_rnn_cell_21/TanhTanh6sequential_30/simple_rnn_11/simple_rnn_cell_21/add:z:0*
T0*'
_output_shapes
:?????????225
3sequential_30/simple_rnn_11/simple_rnn_cell_21/Tanh?
9sequential_30/simple_rnn_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2;
9sequential_30/simple_rnn_11/TensorArrayV2_1/element_shape?
+sequential_30/simple_rnn_11/TensorArrayV2_1TensorListReserveBsequential_30/simple_rnn_11/TensorArrayV2_1/element_shape:output:04sequential_30/simple_rnn_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+sequential_30/simple_rnn_11/TensorArrayV2_1?
 sequential_30/simple_rnn_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_30/simple_rnn_11/time?
4sequential_30/simple_rnn_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4sequential_30/simple_rnn_11/while/maximum_iterations?
.sequential_30/simple_rnn_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_30/simple_rnn_11/while/loop_counter?
!sequential_30/simple_rnn_11/whileWhile7sequential_30/simple_rnn_11/while/loop_counter:output:0=sequential_30/simple_rnn_11/while/maximum_iterations:output:0)sequential_30/simple_rnn_11/time:output:04sequential_30/simple_rnn_11/TensorArrayV2_1:handle:0*sequential_30/simple_rnn_11/zeros:output:04sequential_30/simple_rnn_11/strided_slice_1:output:0Ssequential_30/simple_rnn_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_30_simple_rnn_11_simple_rnn_cell_21_matmul_readvariableop_resourceNsequential_30_simple_rnn_11_simple_rnn_cell_21_biasadd_readvariableop_resourceOsequential_30_simple_rnn_11_simple_rnn_cell_21_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*9
body1R/
-sequential_30_simple_rnn_11_while_body_240701*9
cond1R/
-sequential_30_simple_rnn_11_while_cond_240700*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2#
!sequential_30/simple_rnn_11/while?
Lsequential_30/simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2N
Lsequential_30/simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape?
>sequential_30/simple_rnn_11/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_30/simple_rnn_11/while:output:3Usequential_30/simple_rnn_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype02@
>sequential_30/simple_rnn_11/TensorArrayV2Stack/TensorListStack?
1sequential_30/simple_rnn_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1sequential_30/simple_rnn_11/strided_slice_3/stack?
3sequential_30/simple_rnn_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_30/simple_rnn_11/strided_slice_3/stack_1?
3sequential_30/simple_rnn_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_30/simple_rnn_11/strided_slice_3/stack_2?
+sequential_30/simple_rnn_11/strided_slice_3StridedSliceGsequential_30/simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_30/simple_rnn_11/strided_slice_3/stack:output:0<sequential_30/simple_rnn_11/strided_slice_3/stack_1:output:0<sequential_30/simple_rnn_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2-
+sequential_30/simple_rnn_11/strided_slice_3?
,sequential_30/simple_rnn_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,sequential_30/simple_rnn_11/transpose_1/perm?
'sequential_30/simple_rnn_11/transpose_1	TransposeGsequential_30/simple_rnn_11/TensorArrayV2Stack/TensorListStack:tensor:05sequential_30/simple_rnn_11/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22)
'sequential_30/simple_rnn_11/transpose_1?
 sequential_30/activation_26/ReluRelu4sequential_30/simple_rnn_11/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22"
 sequential_30/activation_26/Relu?
!sequential_30/dropout_11/IdentityIdentity.sequential_30/activation_26/Relu:activations:0*
T0*'
_output_shapes
:?????????22#
!sequential_30/dropout_11/Identity?
/sequential_30/dense_35/MLCMatMul/ReadVariableOpReadVariableOp8sequential_30_dense_35_mlcmatmul_readvariableop_resource*
_output_shapes

:2H*
dtype021
/sequential_30/dense_35/MLCMatMul/ReadVariableOp?
 sequential_30/dense_35/MLCMatMul	MLCMatMul*sequential_30/dropout_11/Identity:output:07sequential_30/dense_35/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2"
 sequential_30/dense_35/MLCMatMul?
-sequential_30/dense_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_30_dense_35_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-sequential_30/dense_35/BiasAdd/ReadVariableOp?
sequential_30/dense_35/BiasAddBiasAdd*sequential_30/dense_35/MLCMatMul:product:05sequential_30/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2 
sequential_30/dense_35/BiasAdd?
sequential_30/dense_35/TanhTanh'sequential_30/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
sequential_30/dense_35/Tanh?
IdentityIdentitysequential_30/dense_35/Tanh:y:0.^sequential_30/dense_35/BiasAdd/ReadVariableOp0^sequential_30/dense_35/MLCMatMul/ReadVariableOpF^sequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOpE^sequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOpG^sequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp"^sequential_30/simple_rnn_10/whileF^sequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOpE^sequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOpG^sequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp"^sequential_30/simple_rnn_11/while*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2^
-sequential_30/dense_35/BiasAdd/ReadVariableOp-sequential_30/dense_35/BiasAdd/ReadVariableOp2b
/sequential_30/dense_35/MLCMatMul/ReadVariableOp/sequential_30/dense_35/MLCMatMul/ReadVariableOp2?
Esequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOpEsequential_30/simple_rnn_10/simple_rnn_cell_20/BiasAdd/ReadVariableOp2?
Dsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOpDsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul/ReadVariableOp2?
Fsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOpFsequential_30/simple_rnn_10/simple_rnn_cell_20/MatMul_1/ReadVariableOp2F
!sequential_30/simple_rnn_10/while!sequential_30/simple_rnn_10/while2?
Esequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOpEsequential_30/simple_rnn_11/simple_rnn_cell_21/BiasAdd/ReadVariableOp2?
Dsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOpDsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul/ReadVariableOp2?
Fsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOpFsequential_30/simple_rnn_11/simple_rnn_cell_21/MatMul_1/ReadVariableOp2F
!sequential_30/simple_rnn_11/while!sequential_30/simple_rnn_11/while:a ]
,
_output_shapes
:??????????
-
_user_specified_namesimple_rnn_10_input
?
?
.__inference_simple_rnn_10_layer_call_fn_243320
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_2411622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?H
?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_241915

inputs5
1simple_rnn_cell_20_matmul_readvariableop_resource6
2simple_rnn_cell_20_biasadd_readvariableop_resource7
3simple_rnn_cell_20_matmul_1_readvariableop_resource
identity??)simple_rnn_cell_20/BiasAdd/ReadVariableOp?(simple_rnn_cell_20/MatMul/ReadVariableOp?*simple_rnn_cell_20/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
(simple_rnn_cell_20/MatMul/ReadVariableOpReadVariableOp1simple_rnn_cell_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(simple_rnn_cell_20/MatMul/ReadVariableOp?
simple_rnn_cell_20/MatMul	MLCMatMulstrided_slice_2:output:00simple_rnn_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul?
)simple_rnn_cell_20/BiasAdd/ReadVariableOpReadVariableOp2simple_rnn_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simple_rnn_cell_20/BiasAdd/ReadVariableOp?
simple_rnn_cell_20/BiasAddBiasAdd#simple_rnn_cell_20/MatMul:product:01simple_rnn_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/BiasAdd?
*simple_rnn_cell_20/MatMul_1/ReadVariableOpReadVariableOp3simple_rnn_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*simple_rnn_cell_20/MatMul_1/ReadVariableOp?
simple_rnn_cell_20/MatMul_1	MLCMatMulzeros:output:02simple_rnn_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/MatMul_1?
simple_rnn_cell_20/addAddV2#simple_rnn_cell_20/BiasAdd:output:0%simple_rnn_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/add?
simple_rnn_cell_20/TanhTanhsimple_rnn_cell_20/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_20/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01simple_rnn_cell_20_matmul_readvariableop_resource2simple_rnn_cell_20_biasadd_readvariableop_resource3simple_rnn_cell_20_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_241849*
condR
while_cond_241848*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1?
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_20/BiasAdd/ReadVariableOp)^simple_rnn_cell_20/MatMul/ReadVariableOp+^simple_rnn_cell_20/MatMul_1/ReadVariableOp^while*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::2V
)simple_rnn_cell_20/BiasAdd/ReadVariableOp)simple_rnn_cell_20/BiasAdd/ReadVariableOp2T
(simple_rnn_cell_20/MatMul/ReadVariableOp(simple_rnn_cell_20/MatMul/ReadVariableOp2X
*simple_rnn_cell_20/MatMul_1/ReadVariableOp*simple_rnn_cell_20/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
X
simple_rnn_10_inputA
%serving_default_simple_rnn_10_input:0??????????<
dense_350
StatefulPartitionedCall:0?????????Htensorflow/serving/predict:??
?7
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?4
_tf_keras_sequential?3{"class_name": "Sequential", "name": "sequential_30", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_30", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_10_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_10", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_25", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_11", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_26", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 72, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 144, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_30", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_10_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_10", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_25", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_11", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_26", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 72, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_rnn_layer?	{"class_name": "SimpleRNN", "name": "simple_rnn_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_10", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 144, 5]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_25", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
cell

state_spec
regularization_losses
	variables
 trainable_variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_rnn_layer?	{"class_name": "SimpleRNN", "name": "simple_rnn_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_11", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 150]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 144, 150]}}
?
"regularization_losses
#	variables
$trainable_variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_26", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
&regularization_losses
'	variables
(trainable_variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 72, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
0iter

1beta_1

2beta_2
	3decay
4learning_rate*m?+m?5m?6m?7m?8m?9m?:m?*v?+v?5v?6v?7v?8v?9v?:v?"
	optimizer
 "
trackable_list_wrapper
X
50
61
72
83
94
:5
*6
+7"
trackable_list_wrapper
X
50
61
72
83
94
:5
*6
+7"
trackable_list_wrapper
?
	regularization_losses
;layer_regularization_losses
<layer_metrics

	variables
=non_trainable_variables
trainable_variables
>metrics

?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

5kernel
6recurrent_kernel
7bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_20", "trainable": true, "dtype": "float32", "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
?
regularization_losses
Dlayer_regularization_losses
Elayer_metrics
	variables
Fnon_trainable_variables
trainable_variables
Gmetrics

Hlayers

Istates
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Jlayer_regularization_losses
Klayer_metrics
	variables
Lnon_trainable_variables
trainable_variables
Mmetrics

Nlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Olayer_regularization_losses
Player_metrics
	variables
Qnon_trainable_variables
trainable_variables
Rmetrics

Slayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

8kernel
9recurrent_kernel
:bias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_21", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
?
regularization_losses
Xlayer_regularization_losses
Ylayer_metrics
	variables
Znon_trainable_variables
 trainable_variables
[metrics

\layers

]states
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"regularization_losses
^layer_regularization_losses
_layer_metrics
#	variables
`non_trainable_variables
$trainable_variables
ametrics

blayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
&regularization_losses
clayer_regularization_losses
dlayer_metrics
'	variables
enon_trainable_variables
(trainable_variables
fmetrics

glayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2H2dense_35/kernel
:H2dense_35/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
,regularization_losses
hlayer_regularization_losses
ilayer_metrics
-	variables
jnon_trainable_variables
.trainable_variables
kmetrics

llayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
::8	?2'simple_rnn_10/simple_rnn_cell_20/kernel
E:C
??21simple_rnn_10/simple_rnn_cell_20/recurrent_kernel
4:2?2%simple_rnn_10/simple_rnn_cell_20/bias
::8	?22'simple_rnn_11/simple_rnn_cell_21/kernel
C:A2221simple_rnn_11/simple_rnn_cell_21/recurrent_kernel
3:122%simple_rnn_11/simple_rnn_cell_21/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
m0
n1
o2"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
?
@regularization_losses
player_regularization_losses
qlayer_metrics
A	variables
rnon_trainable_variables
Btrainable_variables
smetrics

tlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
?
Tregularization_losses
ulayer_regularization_losses
vlayer_metrics
U	variables
wnon_trainable_variables
Vtrainable_variables
xmetrics

ylayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	ztotal
	{count
|	variables
}	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	~total
	count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
z0
{1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
~0
1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
&:$2H2Adam/dense_35/kernel/m
 :H2Adam/dense_35/bias/m
?:=	?2.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/m
J:H
??28Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/m
9:7?2,Adam/simple_rnn_10/simple_rnn_cell_20/bias/m
?:=	?22.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/m
H:F2228Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/m
8:622,Adam/simple_rnn_11/simple_rnn_cell_21/bias/m
&:$2H2Adam/dense_35/kernel/v
 :H2Adam/dense_35/bias/v
?:=	?2.Adam/simple_rnn_10/simple_rnn_cell_20/kernel/v
J:H
??28Adam/simple_rnn_10/simple_rnn_cell_20/recurrent_kernel/v
9:7?2,Adam/simple_rnn_10/simple_rnn_cell_20/bias/v
?:=	?22.Adam/simple_rnn_11/simple_rnn_cell_21/kernel/v
H:F2228Adam/simple_rnn_11/simple_rnn_cell_21/recurrent_kernel/v
8:622,Adam/simple_rnn_11/simple_rnn_cell_21/bias/v
?2?
.__inference_sequential_30_layer_call_fn_242494
.__inference_sequential_30_layer_call_fn_243064
.__inference_sequential_30_layer_call_fn_243085
.__inference_sequential_30_layer_call_fn_242542?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_240776?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/
simple_rnn_10_input??????????
?2?
I__inference_sequential_30_layer_call_and_return_conditional_losses_242445
I__inference_sequential_30_layer_call_and_return_conditional_losses_243043
I__inference_sequential_30_layer_call_and_return_conditional_losses_242418
I__inference_sequential_30_layer_call_and_return_conditional_losses_242812?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_simple_rnn_10_layer_call_fn_243577
.__inference_simple_rnn_10_layer_call_fn_243566
.__inference_simple_rnn_10_layer_call_fn_243320
.__inference_simple_rnn_10_layer_call_fn_243331?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243555
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243309
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243443
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243197?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_activation_25_layer_call_fn_243587?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_25_layer_call_and_return_conditional_losses_243582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_10_layer_call_fn_243611
+__inference_dropout_10_layer_call_fn_243606?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_10_layer_call_and_return_conditional_losses_243601
F__inference_dropout_10_layer_call_and_return_conditional_losses_243596?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_simple_rnn_11_layer_call_fn_243857
.__inference_simple_rnn_11_layer_call_fn_244092
.__inference_simple_rnn_11_layer_call_fn_243846
.__inference_simple_rnn_11_layer_call_fn_244103?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_243969
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_244081
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_243835
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_243723?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_activation_26_layer_call_fn_244113?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_26_layer_call_and_return_conditional_losses_244108?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_11_layer_call_fn_244132
+__inference_dropout_11_layer_call_fn_244137?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_11_layer_call_and_return_conditional_losses_244122
F__inference_dropout_11_layer_call_and_return_conditional_losses_244127?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_35_layer_call_fn_244157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_35_layer_call_and_return_conditional_losses_244148?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_242573simple_rnn_10_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_simple_rnn_cell_20_layer_call_fn_244205
3__inference_simple_rnn_cell_20_layer_call_fn_244219?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_244174
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_244191?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_simple_rnn_cell_21_layer_call_fn_244267
3__inference_simple_rnn_cell_21_layer_call_fn_244281?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_244253
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_244236?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_240776?5768:9*+A?>
7?4
2?/
simple_rnn_10_input??????????
? "3?0
.
dense_35"?
dense_35?????????H?
I__inference_activation_25_layer_call_and_return_conditional_losses_243582d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
.__inference_activation_25_layer_call_fn_243587W5?2
+?(
&?#
inputs???????????
? "?????????????
I__inference_activation_26_layer_call_and_return_conditional_losses_244108X/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? }
.__inference_activation_26_layer_call_fn_244113K/?,
%?"
 ?
inputs?????????2
? "??????????2?
D__inference_dense_35_layer_call_and_return_conditional_losses_244148\*+/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????H
? |
)__inference_dense_35_layer_call_fn_244157O*+/?,
%?"
 ?
inputs?????????2
? "??????????H?
F__inference_dropout_10_layer_call_and_return_conditional_losses_243596h9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
F__inference_dropout_10_layer_call_and_return_conditional_losses_243601h9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
+__inference_dropout_10_layer_call_fn_243606[9?6
/?,
&?#
inputs???????????
p
? "?????????????
+__inference_dropout_10_layer_call_fn_243611[9?6
/?,
&?#
inputs???????????
p 
? "?????????????
F__inference_dropout_11_layer_call_and_return_conditional_losses_244122\3?0
)?&
 ?
inputs?????????2
p
? "%?"
?
0?????????2
? ?
F__inference_dropout_11_layer_call_and_return_conditional_losses_244127\3?0
)?&
 ?
inputs?????????2
p 
? "%?"
?
0?????????2
? ~
+__inference_dropout_11_layer_call_fn_244132O3?0
)?&
 ?
inputs?????????2
p
? "??????????2~
+__inference_dropout_11_layer_call_fn_244137O3?0
)?&
 ?
inputs?????????2
p 
? "??????????2?
I__inference_sequential_30_layer_call_and_return_conditional_losses_242418|5768:9*+I?F
??<
2?/
simple_rnn_10_input??????????
p

 
? "%?"
?
0?????????H
? ?
I__inference_sequential_30_layer_call_and_return_conditional_losses_242445|5768:9*+I?F
??<
2?/
simple_rnn_10_input??????????
p 

 
? "%?"
?
0?????????H
? ?
I__inference_sequential_30_layer_call_and_return_conditional_losses_242812o5768:9*+<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????H
? ?
I__inference_sequential_30_layer_call_and_return_conditional_losses_243043o5768:9*+<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????H
? ?
.__inference_sequential_30_layer_call_fn_242494o5768:9*+I?F
??<
2?/
simple_rnn_10_input??????????
p

 
? "??????????H?
.__inference_sequential_30_layer_call_fn_242542o5768:9*+I?F
??<
2?/
simple_rnn_10_input??????????
p 

 
? "??????????H?
.__inference_sequential_30_layer_call_fn_243064b5768:9*+<?9
2?/
%?"
inputs??????????
p

 
? "??????????H?
.__inference_sequential_30_layer_call_fn_243085b5768:9*+<?9
2?/
%?"
inputs??????????
p 

 
? "??????????H?
$__inference_signature_wrapper_242573?5768:9*+X?U
? 
N?K
I
simple_rnn_10_input2?/
simple_rnn_10_input??????????"3?0
.
dense_35"?
dense_35?????????H?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243197?576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243309?576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243443t576@?=
6?3
%?"
inputs??????????

 
p

 
? "+?(
!?
0???????????
? ?
I__inference_simple_rnn_10_layer_call_and_return_conditional_losses_243555t576@?=
6?3
%?"
inputs??????????

 
p 

 
? "+?(
!?
0???????????
? ?
.__inference_simple_rnn_10_layer_call_fn_243320~576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "&?#????????????????????
.__inference_simple_rnn_10_layer_call_fn_243331~576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "&?#????????????????????
.__inference_simple_rnn_10_layer_call_fn_243566g576@?=
6?3
%?"
inputs??????????

 
p

 
? "?????????????
.__inference_simple_rnn_10_layer_call_fn_243577g576@?=
6?3
%?"
inputs??????????

 
p 

 
? "?????????????
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_243723o8:9A?>
7?4
&?#
inputs???????????

 
p

 
? "%?"
?
0?????????2
? ?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_243835o8:9A?>
7?4
&?#
inputs???????????

 
p 

 
? "%?"
?
0?????????2
? ?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_243969~8:9P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "%?"
?
0?????????2
? ?
I__inference_simple_rnn_11_layer_call_and_return_conditional_losses_244081~8:9P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "%?"
?
0?????????2
? ?
.__inference_simple_rnn_11_layer_call_fn_243846b8:9A?>
7?4
&?#
inputs???????????

 
p

 
? "??????????2?
.__inference_simple_rnn_11_layer_call_fn_243857b8:9A?>
7?4
&?#
inputs???????????

 
p 

 
? "??????????2?
.__inference_simple_rnn_11_layer_call_fn_244092q8:9P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "??????????2?
.__inference_simple_rnn_11_layer_call_fn_244103q8:9P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "??????????2?
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_244174?576]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
N__inference_simple_rnn_cell_20_layer_call_and_return_conditional_losses_244191?576]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
3__inference_simple_rnn_cell_20_layer_call_fn_244205?576]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "F?C
?
0??????????
#? 
?
1/0???????????
3__inference_simple_rnn_cell_20_layer_call_fn_244219?576]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "F?C
?
0??????????
#? 
?
1/0???????????
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_244236?8:9]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????2
p
? "R?O
H?E
?
0/0?????????2
$?!
?
0/1/0?????????2
? ?
N__inference_simple_rnn_cell_21_layer_call_and_return_conditional_losses_244253?8:9]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????2
p 
? "R?O
H?E
?
0/0?????????2
$?!
?
0/1/0?????????2
? ?
3__inference_simple_rnn_cell_21_layer_call_fn_244267?8:9]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????2
p
? "D?A
?
0?????????2
"?
?
1/0?????????2?
3__inference_simple_rnn_cell_21_layer_call_fn_244281?8:9]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????2
p 
? "D?A
?
0?????????2
"?
?
1/0?????????2