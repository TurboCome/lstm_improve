ߘ"
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
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	2?*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
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
%simple_rnn_2/simple_rnn_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*6
shared_name'%simple_rnn_2/simple_rnn_cell_4/kernel
?
9simple_rnn_2/simple_rnn_cell_4/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_2/simple_rnn_cell_4/kernel*
_output_shapes
:	?*
dtype0
?
/simple_rnn_2/simple_rnn_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*@
shared_name1/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel
?
Csimple_rnn_2/simple_rnn_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
#simple_rnn_2/simple_rnn_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#simple_rnn_2/simple_rnn_cell_4/bias
?
7simple_rnn_2/simple_rnn_cell_4/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_2/simple_rnn_cell_4/bias*
_output_shapes	
:?*
dtype0
?
%simple_rnn_3/simple_rnn_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*6
shared_name'%simple_rnn_3/simple_rnn_cell_5/kernel
?
9simple_rnn_3/simple_rnn_cell_5/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_3/simple_rnn_cell_5/kernel*
_output_shapes
:	?2*
dtype0
?
/simple_rnn_3/simple_rnn_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*@
shared_name1/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel
?
Csimple_rnn_3/simple_rnn_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel*
_output_shapes

:22*
dtype0
?
#simple_rnn_3/simple_rnn_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#simple_rnn_3/simple_rnn_cell_5/bias
?
7simple_rnn_3/simple_rnn_cell_5/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_3/simple_rnn_cell_5/bias*
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
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	2?*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*=
shared_name.,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m
?
@Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m*
_output_shapes
:	?*
dtype0
?
6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*G
shared_name86Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m
?
JAdam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
*Adam/simple_rnn_2/simple_rnn_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/simple_rnn_2/simple_rnn_cell_4/bias/m
?
>Adam/simple_rnn_2/simple_rnn_cell_4/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_2/simple_rnn_cell_4/bias/m*
_output_shapes	
:?*
dtype0
?
,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*=
shared_name.,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/m
?
@Adam/simple_rnn_3/simple_rnn_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/m*
_output_shapes
:	?2*
dtype0
?
6Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*G
shared_name86Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/m
?
JAdam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/m*
_output_shapes

:22*
dtype0
?
*Adam/simple_rnn_3/simple_rnn_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/simple_rnn_3/simple_rnn_cell_5/bias/m
?
>Adam/simple_rnn_3/simple_rnn_cell_5/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_3/simple_rnn_cell_5/bias/m*
_output_shapes
:2*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	2?*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*=
shared_name.,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v
?
@Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v*
_output_shapes
:	?*
dtype0
?
6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*G
shared_name86Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v
?
JAdam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
*Adam/simple_rnn_2/simple_rnn_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*Adam/simple_rnn_2/simple_rnn_cell_4/bias/v
?
>Adam/simple_rnn_2/simple_rnn_cell_4/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_2/simple_rnn_cell_4/bias/v*
_output_shapes	
:?*
dtype0
?
,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*=
shared_name.,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/v
?
@Adam/simple_rnn_3/simple_rnn_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/v*
_output_shapes
:	?2*
dtype0
?
6Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*G
shared_name86Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/v
?
JAdam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/v*
_output_shapes

:22*
dtype0
?
*Adam/simple_rnn_3/simple_rnn_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/simple_rnn_3/simple_rnn_cell_5/bias/v
?
>Adam/simple_rnn_3/simple_rnn_cell_5/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_3/simple_rnn_cell_5/bias/v*
_output_shapes
:2*
dtype0

NoOpNoOp
?>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?>
value?>B?> B?=
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
		variables

trainable_variables
regularization_losses
	keras_api

signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
 regularization_losses
!	keras_api
R
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
?
0iter

1beta_1

2beta_2
	3decay
4learning_rate*m?+m?5m?6m?7m?8m?9m?:m?*v?+v?5v?6v?7v?8v?9v?:v?
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
 
?
;metrics
		variables

trainable_variables
regularization_losses
<layer_regularization_losses
=non_trainable_variables
>layer_metrics

?layers
 
~

5kernel
6recurrent_kernel
7bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
 

50
61
72

50
61
72
 
?
Dmetrics

Estates
	variables
trainable_variables
regularization_losses
Flayer_regularization_losses
Gnon_trainable_variables
Hlayer_metrics

Ilayers
 
 
 
?
Jmetrics
	variables
trainable_variables
regularization_losses

Klayers
Llayer_regularization_losses
Mlayer_metrics
Nnon_trainable_variables
 
 
 
?
Ometrics
	variables
trainable_variables
regularization_losses

Players
Qlayer_regularization_losses
Rlayer_metrics
Snon_trainable_variables
~

8kernel
9recurrent_kernel
:bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
 

80
91
:2

80
91
:2
 
?
Xmetrics

Ystates
	variables
trainable_variables
 regularization_losses
Zlayer_regularization_losses
[non_trainable_variables
\layer_metrics

]layers
 
 
 
?
^metrics
"	variables
#trainable_variables
$regularization_losses

_layers
`layer_regularization_losses
alayer_metrics
bnon_trainable_variables
 
 
 
?
cmetrics
&	variables
'trainable_variables
(regularization_losses

dlayers
elayer_regularization_losses
flayer_metrics
gnon_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
?
hmetrics
,	variables
-trainable_variables
.regularization_losses

ilayers
jlayer_regularization_losses
klayer_metrics
lnon_trainable_variables
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
a_
VARIABLE_VALUE%simple_rnn_2/simple_rnn_cell_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_2/simple_rnn_cell_4/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%simple_rnn_3/simple_rnn_cell_5/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_3/simple_rnn_cell_5/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE

m0
n1
o2
 
 
 
1
0
1
2
3
4
5
6

50
61
72

50
61
72
 
?
pmetrics
@	variables
Atrainable_variables
Bregularization_losses

qlayers
rlayer_regularization_losses
slayer_metrics
tnon_trainable_variables
 
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

80
91
:2

80
91
:2
 
?
umetrics
T	variables
Utrainable_variables
Vregularization_losses

vlayers
wlayer_regularization_losses
xlayer_metrics
ynon_trainable_variables
 
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
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_2/simple_rnn_cell_4/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_3/simple_rnn_cell_5/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_2/simple_rnn_cell_4/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_3/simple_rnn_cell_5/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
"serving_default_simple_rnn_2_inputPlaceholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_2_input%simple_rnn_2/simple_rnn_cell_4/kernel#simple_rnn_2/simple_rnn_cell_4/bias/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel%simple_rnn_3/simple_rnn_cell_5/kernel#simple_rnn_3/simple_rnn_cell_5/bias/simple_rnn_3/simple_rnn_cell_5/recurrent_kerneldense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_20572
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_2/simple_rnn_cell_4/kernel/Read/ReadVariableOpCsimple_rnn_2/simple_rnn_cell_4/recurrent_kernel/Read/ReadVariableOp7simple_rnn_2/simple_rnn_cell_4/bias/Read/ReadVariableOp9simple_rnn_3/simple_rnn_cell_5/kernel/Read/ReadVariableOpCsimple_rnn_3/simple_rnn_cell_5/recurrent_kernel/Read/ReadVariableOp7simple_rnn_3/simple_rnn_cell_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp@Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_2/simple_rnn_cell_4/bias/m/Read/ReadVariableOp@Adam/simple_rnn_3/simple_rnn_cell_5/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_3/simple_rnn_cell_5/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp@Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_2/simple_rnn_cell_4/bias/v/Read/ReadVariableOp@Adam/simple_rnn_3/simple_rnn_cell_5/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_3/simple_rnn_cell_5/bias/v/Read/ReadVariableOpConst*0
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
GPU 2J 8? *'
f"R 
__inference__traced_save_22408
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_2/simple_rnn_cell_4/kernel/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel#simple_rnn_2/simple_rnn_cell_4/bias%simple_rnn_3/simple_rnn_cell_5/kernel/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel#simple_rnn_3/simple_rnn_cell_5/biastotalcounttotal_1count_1total_2count_2Adam/dense_1/kernel/mAdam/dense_1/bias/m,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m*Adam/simple_rnn_2/simple_rnn_cell_4/bias/m,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/m6Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/m*Adam/simple_rnn_3/simple_rnn_cell_5/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/v,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v6Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v*Adam/simple_rnn_2/simple_rnn_cell_4/bias/v,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/v6Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/v*Adam/simple_rnn_3/simple_rnn_cell_5/bias/v*/
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_22523??
?
?
while_cond_21901
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21901___redundant_placeholder03
/while_while_cond_21901___redundant_placeholder13
/while_while_cond_21901___redundant_placeholder23
/while_while_cond_21901___redundant_placeholder3
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
while_cond_21241
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21241___redundant_placeholder03
/while_while_cond_21241___redundant_placeholder13
/while_while_cond_21241___redundant_placeholder23
/while_while_cond_21241___redundant_placeholder3
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
?3
?
while_body_21242
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource??.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_4/MatMul/ReadVariableOp?/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp?
while/simple_rnn_cell_4/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_4/MatMul?
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_4/BiasAdd?
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_4/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_4/MatMul_1?
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/add?
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_22107

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
?
?
#__inference_signature_wrapper_20572
simple_rnn_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_187752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
,
_output_shapes
:??????????
,
_user_specified_namesimple_rnn_2_input
?<
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_19161

inputs
simple_rnn_cell_4_19086
simple_rnn_cell_4_19088
simple_rnn_cell_4_19090
identity??)simple_rnn_cell_4/StatefulPartitionedCall?whileD
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
)simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_4_19086simple_rnn_cell_4_19088simple_rnn_cell_4_19090*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_188242+
)simple_rnn_cell_4/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_4_19086simple_rnn_cell_4_19088simple_rnn_cell_4_19090*
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
bodyR
while_body_19098*
condR
while_cond_19097*9
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
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_4/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_4/StatefulPartitionedCall)simple_rnn_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?

?
simple_rnn_3_while_cond_207316
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_28
4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1M
Isimple_rnn_3_while_simple_rnn_3_while_cond_20731___redundant_placeholder0M
Isimple_rnn_3_while_simple_rnn_3_while_cond_20731___redundant_placeholder1M
Isimple_rnn_3_while_simple_rnn_3_while_cond_20731___redundant_placeholder2M
Isimple_rnn_3_while_simple_rnn_3_while_cond_20731___redundant_placeholder3
simple_rnn_3_while_identity
?
simple_rnn_3/while/LessLesssimple_rnn_3_while_placeholder4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_3/while/Less?
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_3/while/Identity"C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0*@
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
?G
?
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_21834

inputs4
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_5/BiasAdd/ReadVariableOp?'simple_rnn_cell_5/MatMul/ReadVariableOp?)simple_rnn_cell_5/MatMul_1/ReadVariableOp?whileD
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
:???????????2
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
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp?
simple_rnn_cell_5/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul?
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp?
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/BiasAdd?
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp?
simple_rnn_cell_5/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul_1?
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/add?
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
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
bodyR
while_body_21768*
condR
while_cond_21767*8
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
:??????????2*
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
:??????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?Q
?
*sequential_1_simple_rnn_3_while_body_18700P
Lsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_loop_counterV
Rsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_maximum_iterations/
+sequential_1_simple_rnn_3_while_placeholder1
-sequential_1_simple_rnn_3_while_placeholder_11
-sequential_1_simple_rnn_3_while_placeholder_2O
Ksequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_strided_slice_1_0?
?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0V
Rsequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resource_0W
Ssequential_1_simple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0X
Tsequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0,
(sequential_1_simple_rnn_3_while_identity.
*sequential_1_simple_rnn_3_while_identity_1.
*sequential_1_simple_rnn_3_while_identity_2.
*sequential_1_simple_rnn_3_while_identity_3.
*sequential_1_simple_rnn_3_while_identity_4M
Isequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_strided_slice_1?
?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorT
Psequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resourceU
Qsequential_1_simple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resourceV
Rsequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource??Hsequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?Gsequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp?Isequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
Qsequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2S
Qsequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Csequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0+sequential_1_simple_rnn_3_while_placeholderZsequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02E
Csequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem?
Gsequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpRsequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02I
Gsequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp?
8sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul	MLCMatMulJsequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22:
8sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul?
Hsequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpSsequential_1_simple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02J
Hsequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
9sequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAddBiasAddBsequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul:product:0Psequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22;
9sequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd?
Isequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpTsequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02K
Isequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
:sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1	MLCMatMul-sequential_1_simple_rnn_3_while_placeholder_2Qsequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22<
:sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1?
5sequential_1/simple_rnn_3/while/simple_rnn_cell_5/addAddV2Bsequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd:output:0Dsequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????227
5sequential_1/simple_rnn_3/while/simple_rnn_cell_5/add?
6sequential_1/simple_rnn_3/while/simple_rnn_cell_5/TanhTanh9sequential_1/simple_rnn_3/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????228
6sequential_1/simple_rnn_3/while/simple_rnn_cell_5/Tanh?
Dsequential_1/simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_1_simple_rnn_3_while_placeholder_1+sequential_1_simple_rnn_3_while_placeholder:sequential_1/simple_rnn_3/while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype02F
Dsequential_1/simple_rnn_3/while/TensorArrayV2Write/TensorListSetItem?
%sequential_1/simple_rnn_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/simple_rnn_3/while/add/y?
#sequential_1/simple_rnn_3/while/addAddV2+sequential_1_simple_rnn_3_while_placeholder.sequential_1/simple_rnn_3/while/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/simple_rnn_3/while/add?
'sequential_1/simple_rnn_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/simple_rnn_3/while/add_1/y?
%sequential_1/simple_rnn_3/while/add_1AddV2Lsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_loop_counter0sequential_1/simple_rnn_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/simple_rnn_3/while/add_1?
(sequential_1/simple_rnn_3/while/IdentityIdentity)sequential_1/simple_rnn_3/while/add_1:z:0I^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_1/simple_rnn_3/while/Identity?
*sequential_1/simple_rnn_3/while/Identity_1IdentityRsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_maximum_iterationsI^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_1/simple_rnn_3/while/Identity_1?
*sequential_1/simple_rnn_3/while/Identity_2Identity'sequential_1/simple_rnn_3/while/add:z:0I^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_1/simple_rnn_3/while/Identity_2?
*sequential_1/simple_rnn_3/while/Identity_3IdentityTsequential_1/simple_rnn_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0I^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_1/simple_rnn_3/while/Identity_3?
*sequential_1/simple_rnn_3/while/Identity_4Identity:sequential_1/simple_rnn_3/while/simple_rnn_cell_5/Tanh:y:0I^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22,
*sequential_1/simple_rnn_3/while/Identity_4"]
(sequential_1_simple_rnn_3_while_identity1sequential_1/simple_rnn_3/while/Identity:output:0"a
*sequential_1_simple_rnn_3_while_identity_13sequential_1/simple_rnn_3/while/Identity_1:output:0"a
*sequential_1_simple_rnn_3_while_identity_23sequential_1/simple_rnn_3/while/Identity_2:output:0"a
*sequential_1_simple_rnn_3_while_identity_33sequential_1/simple_rnn_3/while/Identity_3:output:0"a
*sequential_1_simple_rnn_3_while_identity_43sequential_1/simple_rnn_3/while/Identity_4:output:0"?
Isequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_strided_slice_1Ksequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_strided_slice_1_0"?
Qsequential_1_simple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resourceSsequential_1_simple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"?
Rsequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceTsequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"?
Psequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resourceRsequential_1_simple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"?
?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2?
Hsequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpHsequential_1/simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2?
Gsequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOpGsequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp2?
Isequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpIsequential_1/simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
?
E
)__inference_dropout_3_layer_call_fn_22136

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
GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_203762
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
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21554
inputs_04
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_4/BiasAdd/ReadVariableOp?'simple_rnn_cell_4/MatMul/ReadVariableOp?)simple_rnn_cell_4/MatMul_1/ReadVariableOp?whileF
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
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp?
simple_rnn_cell_4/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul?
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp?
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/BiasAdd?
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp?
simple_rnn_cell_4/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul_1?
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/add?
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
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
bodyR
while_body_21488*
condR
while_cond_21487*9
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
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
,__inference_simple_rnn_2_layer_call_fn_21565
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_191612
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
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_21968
inputs_04
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_5/BiasAdd/ReadVariableOp?'simple_rnn_cell_5/MatMul/ReadVariableOp?)simple_rnn_cell_5/MatMul_1/ReadVariableOp?whileF
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
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp?
simple_rnn_cell_5/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul?
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp?
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/BiasAdd?
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp?
simple_rnn_cell_5/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul_1?
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/add?
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
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
bodyR
while_body_21902*
condR
while_cond_21901*8
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
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?B
?
simple_rnn_2_while_body_208576
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0J
Fsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0K
Gsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resourceH
Dsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resourceI
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource??;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp?<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02<
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp?
+simple_rnn_2/while/simple_rnn_cell_4/MatMul	MLCMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+simple_rnn_2/while/simple_rnn_cell_4/MatMul?
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02=
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
,simple_rnn_2/while/simple_rnn_cell_4/BiasAddBiasAdd5simple_rnn_2/while/simple_rnn_cell_4/MatMul:product:0Csimple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,simple_rnn_2/while/simple_rnn_cell_4/BiasAdd?
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02>
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
-simple_rnn_2/while/simple_rnn_cell_4/MatMul_1	MLCMatMul simple_rnn_2_while_placeholder_2Dsimple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_2/while/simple_rnn_cell_4/MatMul_1?
(simple_rnn_2/while/simple_rnn_cell_4/addAddV25simple_rnn_2/while/simple_rnn_cell_4/BiasAdd:output:07simple_rnn_2/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_2/while/simple_rnn_cell_4/add?
)simple_rnn_2/while/simple_rnn_cell_4/TanhTanh,simple_rnn_2/while/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_2/while/simple_rnn_cell_4/Tanh?
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1simple_rnn_2_while_placeholder-simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_2/while/add/y?
simple_rnn_2/while/addAddV2simple_rnn_2_while_placeholder!simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/while/addz
simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_2/while/add_1/y?
simple_rnn_2/while/add_1AddV22simple_rnn_2_while_simple_rnn_2_while_loop_counter#simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/while/add_1?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/add_1:z:0<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity?
simple_rnn_2/while/Identity_1Identity8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_1?
simple_rnn_2/while/Identity_2Identitysimple_rnn_2/while/add:z:0<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_2?
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_3?
simple_rnn_2/while/Identity_4Identity-simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
simple_rnn_2/while/Identity_4"C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"?
Dsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resourceFsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"?
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"?
Csimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resourceEsimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"?
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2z
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2x
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
?3
?
while_body_22014
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource??.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_5/MatMul/ReadVariableOp?/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp?
while/simple_rnn_cell_5/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_5/MatMul?
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_5/BiasAdd?
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_5/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_5/MatMul_1?
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/add?
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
?
?
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_22173

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
?
?
while_cond_21767
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21767___redundant_placeholder03
/while_while_cond_21767___redundant_placeholder13
/while_while_cond_21767___redundant_placeholder23
/while_while_cond_21767___redundant_placeholder3
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
while_cond_21129
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21129___redundant_placeholder03
/while_while_cond_21129___redundant_placeholder13
/while_while_cond_21129___redundant_placeholder23
/while_while_cond_21129___redundant_placeholder3
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
?G
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_20026

inputs4
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_4/BiasAdd/ReadVariableOp?'simple_rnn_cell_4/MatMul/ReadVariableOp?)simple_rnn_cell_4/MatMul_1/ReadVariableOp?whileD
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
:??????????2
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
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp?
simple_rnn_cell_4/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul?
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp?
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/BiasAdd?
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp?
simple_rnn_cell_4/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul_1?
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/add?
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
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
bodyR
while_body_19960*
condR
while_cond_19959*9
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
:???????????*
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
:???????????2
transpose_1?
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_21600

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_21605

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
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_200782
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
while_cond_19847
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_19847___redundant_placeholder03
/while_while_cond_19847___redundant_placeholder13
/while_while_cond_19847___redundant_placeholder23
/while_while_cond_19847___redundant_placeholder3
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
?G
?
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_21722

inputs4
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_5/BiasAdd/ReadVariableOp?'simple_rnn_cell_5/MatMul/ReadVariableOp?)simple_rnn_cell_5/MatMul_1/ReadVariableOp?whileD
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
:???????????2
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
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp?
simple_rnn_cell_5/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul?
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp?
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/BiasAdd?
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp?
simple_rnn_cell_5/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul_1?
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/add?
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
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
bodyR
while_body_21656*
condR
while_cond_21655*8
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
:??????????2*
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
:??????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_simple_rnn_3_layer_call_fn_22091
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_196732
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
while_cond_20252
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_20252___redundant_placeholder03
/while_while_cond_20252___redundant_placeholder13
/while_while_cond_20252___redundant_placeholder23
/while_while_cond_20252___redundant_placeholder3
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
?<
?
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_19790

inputs
simple_rnn_cell_5_19715
simple_rnn_cell_5_19717
simple_rnn_cell_5_19719
identity??)simple_rnn_cell_5/StatefulPartitionedCall?whileD
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
)simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_5_19715simple_rnn_cell_5_19717simple_rnn_cell_5_19719*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_193532+
)simple_rnn_cell_5/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_5_19715simple_rnn_cell_5_19717simple_rnn_cell_5_19719*
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
bodyR
while_body_19727*
condR
while_cond_19726*8
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
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2V
)simple_rnn_cell_5/StatefulPartitionedCall)simple_rnn_cell_5/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
,__inference_simple_rnn_3_layer_call_fn_21856

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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_203192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_21042

inputsA
=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resourceB
>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resourceC
?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resourceA
=simple_rnn_3_simple_rnn_cell_5_matmul_readvariableop_resourceB
>simple_rnn_3_simple_rnn_cell_5_biasadd_readvariableop_resourceC
?simple_rnn_3_simple_rnn_cell_5_matmul_1_readvariableop_resource-
)dense_1_mlcmatmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp?4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp?6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp?simple_rnn_2/while?5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp?4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp?6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp?simple_rnn_3/while^
simple_rnn_2/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_2/Shape?
 simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_2/strided_slice/stack?
"simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_2/strided_slice/stack_1?
"simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_2/strided_slice/stack_2?
simple_rnn_2/strided_sliceStridedSlicesimple_rnn_2/Shape:output:0)simple_rnn_2/strided_slice/stack:output:0+simple_rnn_2/strided_slice/stack_1:output:0+simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_2/strided_slicew
simple_rnn_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/mul/y?
simple_rnn_2/zeros/mulMul#simple_rnn_2/strided_slice:output:0!simple_rnn_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/zeros/muly
simple_rnn_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/Less/y?
simple_rnn_2/zeros/LessLesssimple_rnn_2/zeros/mul:z:0"simple_rnn_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/zeros/Less}
simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/packed/1?
simple_rnn_2/zeros/packedPack#simple_rnn_2/strided_slice:output:0$simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_2/zeros/packedy
simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_2/zeros/Const?
simple_rnn_2/zerosFill"simple_rnn_2/zeros/packed:output:0!simple_rnn_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_2/zeros?
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_2/transpose/perm?
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
simple_rnn_2/transposev
simple_rnn_2/Shape_1Shapesimple_rnn_2/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_2/Shape_1?
"simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_2/strided_slice_1/stack?
$simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_1/stack_1?
$simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_1/stack_2?
simple_rnn_2/strided_slice_1StridedSlicesimple_rnn_2/Shape_1:output:0+simple_rnn_2/strided_slice_1/stack:output:0-simple_rnn_2/strided_slice_1/stack_1:output:0-simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_2/strided_slice_1?
(simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_2/TensorArrayV2/element_shape?
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_2/TensorArrayV2?
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_2/strided_slice_2/stack?
$simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_2/stack_1?
$simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_2/stack_2?
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_2/strided_slice_2?
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp?
%simple_rnn_2/simple_rnn_cell_4/MatMul	MLCMatMul%simple_rnn_2/strided_slice_2:output:0<simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%simple_rnn_2/simple_rnn_cell_4/MatMul?
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
&simple_rnn_2/simple_rnn_cell_4/BiasAddBiasAdd/simple_rnn_2/simple_rnn_cell_4/MatMul:product:0=simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&simple_rnn_2/simple_rnn_cell_4/BiasAdd?
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
'simple_rnn_2/simple_rnn_cell_4/MatMul_1	MLCMatMulsimple_rnn_2/zeros:output:0>simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_2/simple_rnn_cell_4/MatMul_1?
"simple_rnn_2/simple_rnn_cell_4/addAddV2/simple_rnn_2/simple_rnn_cell_4/BiasAdd:output:01simple_rnn_2/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2$
"simple_rnn_2/simple_rnn_cell_4/add?
#simple_rnn_2/simple_rnn_cell_4/TanhTanh&simple_rnn_2/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2%
#simple_rnn_2/simple_rnn_cell_4/Tanh?
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2,
*simple_rnn_2/TensorArrayV2_1/element_shape?
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_2/TensorArrayV2_1h
simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_2/time?
%simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_2/while/maximum_iterations?
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_2/while/loop_counter?
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_2_while_body_20857*)
cond!R
simple_rnn_2_while_cond_20856*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_2/while?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype021
/simple_rnn_2/TensorArrayV2Stack/TensorListStack?
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_2/strided_slice_3/stack?
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_2/strided_slice_3/stack_1?
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_3/stack_2?
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_2/strided_slice_3?
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_2/transpose_1/perm?
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
simple_rnn_2/transpose_1?
activation_2/ReluRelusimple_rnn_2/transpose_1:y:0*
T0*-
_output_shapes
:???????????2
activation_2/Relu?
dropout_2/IdentityIdentityactivation_2/Relu:activations:0*
T0*-
_output_shapes
:???????????2
dropout_2/Identitys
simple_rnn_3/ShapeShapedropout_2/Identity:output:0*
T0*
_output_shapes
:2
simple_rnn_3/Shape?
 simple_rnn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_3/strided_slice/stack?
"simple_rnn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_3/strided_slice/stack_1?
"simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_3/strided_slice/stack_2?
simple_rnn_3/strided_sliceStridedSlicesimple_rnn_3/Shape:output:0)simple_rnn_3/strided_slice/stack:output:0+simple_rnn_3/strided_slice/stack_1:output:0+simple_rnn_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_3/strided_slicev
simple_rnn_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_3/zeros/mul/y?
simple_rnn_3/zeros/mulMul#simple_rnn_3/strided_slice:output:0!simple_rnn_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/zeros/muly
simple_rnn_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_3/zeros/Less/y?
simple_rnn_3/zeros/LessLesssimple_rnn_3/zeros/mul:z:0"simple_rnn_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/zeros/Less|
simple_rnn_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_3/zeros/packed/1?
simple_rnn_3/zeros/packedPack#simple_rnn_3/strided_slice:output:0$simple_rnn_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_3/zeros/packedy
simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_3/zeros/Const?
simple_rnn_3/zerosFill"simple_rnn_3/zeros/packed:output:0!simple_rnn_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
simple_rnn_3/zeros?
simple_rnn_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_3/transpose/perm?
simple_rnn_3/transpose	Transposedropout_2/Identity:output:0$simple_rnn_3/transpose/perm:output:0*
T0*-
_output_shapes
:???????????2
simple_rnn_3/transposev
simple_rnn_3/Shape_1Shapesimple_rnn_3/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_3/Shape_1?
"simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_3/strided_slice_1/stack?
$simple_rnn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_1/stack_1?
$simple_rnn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_1/stack_2?
simple_rnn_3/strided_slice_1StridedSlicesimple_rnn_3/Shape_1:output:0+simple_rnn_3/strided_slice_1/stack:output:0-simple_rnn_3/strided_slice_1/stack_1:output:0-simple_rnn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_3/strided_slice_1?
(simple_rnn_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_3/TensorArrayV2/element_shape?
simple_rnn_3/TensorArrayV2TensorListReserve1simple_rnn_3/TensorArrayV2/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_3/TensorArrayV2?
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2D
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_3/transpose:y:0Ksimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_3/strided_slice_2/stack?
$simple_rnn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_2/stack_1?
$simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_2/stack_2?
simple_rnn_3/strided_slice_2StridedSlicesimple_rnn_3/transpose:y:0+simple_rnn_3/strided_slice_2/stack:output:0-simple_rnn_3/strided_slice_2/stack_1:output:0-simple_rnn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_3/strided_slice_2?
4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp=simple_rnn_3_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype026
4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp?
%simple_rnn_3/simple_rnn_cell_5/MatMul	MLCMatMul%simple_rnn_3/strided_slice_2:output:0<simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22'
%simple_rnn_3/simple_rnn_cell_5/MatMul?
5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_3_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype027
5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
&simple_rnn_3/simple_rnn_cell_5/BiasAddBiasAdd/simple_rnn_3/simple_rnn_cell_5/MatMul:product:0=simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22(
&simple_rnn_3/simple_rnn_cell_5/BiasAdd?
6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_3_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype028
6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
'simple_rnn_3/simple_rnn_cell_5/MatMul_1	MLCMatMulsimple_rnn_3/zeros:output:0>simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22)
'simple_rnn_3/simple_rnn_cell_5/MatMul_1?
"simple_rnn_3/simple_rnn_cell_5/addAddV2/simple_rnn_3/simple_rnn_cell_5/BiasAdd:output:01simple_rnn_3/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22$
"simple_rnn_3/simple_rnn_cell_5/add?
#simple_rnn_3/simple_rnn_cell_5/TanhTanh&simple_rnn_3/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22%
#simple_rnn_3/simple_rnn_cell_5/Tanh?
*simple_rnn_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2,
*simple_rnn_3/TensorArrayV2_1/element_shape?
simple_rnn_3/TensorArrayV2_1TensorListReserve3simple_rnn_3/TensorArrayV2_1/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_3/TensorArrayV2_1h
simple_rnn_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_3/time?
%simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_3/while/maximum_iterations?
simple_rnn_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_3/while/loop_counter?
simple_rnn_3/whileWhile(simple_rnn_3/while/loop_counter:output:0.simple_rnn_3/while/maximum_iterations:output:0simple_rnn_3/time:output:0%simple_rnn_3/TensorArrayV2_1:handle:0simple_rnn_3/zeros:output:0%simple_rnn_3/strided_slice_1:output:0Dsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_3_simple_rnn_cell_5_matmul_readvariableop_resource>simple_rnn_3_simple_rnn_cell_5_biasadd_readvariableop_resource?simple_rnn_3_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_3_while_body_20967*)
cond!R
simple_rnn_3_while_cond_20966*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
simple_rnn_3/while?
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2?
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_3/while:output:3Fsimple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype021
/simple_rnn_3/TensorArrayV2Stack/TensorListStack?
"simple_rnn_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_3/strided_slice_3/stack?
$simple_rnn_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_3/strided_slice_3/stack_1?
$simple_rnn_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_3/stack_2?
simple_rnn_3/strided_slice_3StridedSlice8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_3/strided_slice_3/stack:output:0-simple_rnn_3/strided_slice_3/stack_1:output:0-simple_rnn_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
simple_rnn_3/strided_slice_3?
simple_rnn_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_3/transpose_1/perm?
simple_rnn_3/transpose_1	Transpose8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22
simple_rnn_3/transpose_1?
activation_3/ReluRelu%simple_rnn_3/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22
activation_3/Relu?
dropout_3/IdentityIdentityactivation_3/Relu:activations:0*
T0*'
_output_shapes
:?????????22
dropout_3/Identity?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMuldropout_3/Identity:output:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Tanh?
IdentityIdentitydense_1/Tanh:y:0^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp6^simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp5^simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp^simple_rnn_2/while6^simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp5^simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp7^simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp^simple_rnn_3/while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp2n
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp2l
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while2n
5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp2l
4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp2p
6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp2(
simple_rnn_3/whilesimple_rnn_3/while:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_18824

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
?H
?
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_22080
inputs_04
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_5/BiasAdd/ReadVariableOp?'simple_rnn_cell_5/MatMul/ReadVariableOp?)simple_rnn_cell_5/MatMul_1/ReadVariableOp?whileF
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
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp?
simple_rnn_cell_5/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul?
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp?
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/BiasAdd?
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp?
simple_rnn_cell_5/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul_1?
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/add?
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
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
bodyR
while_body_22014*
condR
while_cond_22013*8
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
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_19609
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_19609___redundant_placeholder03
/while_while_cond_19609___redundant_placeholder13
/while_while_cond_19609___redundant_placeholder23
/while_while_cond_19609___redundant_placeholder3
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
?
*sequential_1_simple_rnn_2_while_cond_18589P
Lsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_loop_counterV
Rsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_maximum_iterations/
+sequential_1_simple_rnn_2_while_placeholder1
-sequential_1_simple_rnn_2_while_placeholder_11
-sequential_1_simple_rnn_2_while_placeholder_2R
Nsequential_1_simple_rnn_2_while_less_sequential_1_simple_rnn_2_strided_slice_1g
csequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_cond_18589___redundant_placeholder0g
csequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_cond_18589___redundant_placeholder1g
csequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_cond_18589___redundant_placeholder2g
csequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_cond_18589___redundant_placeholder3,
(sequential_1_simple_rnn_2_while_identity
?
$sequential_1/simple_rnn_2/while/LessLess+sequential_1_simple_rnn_2_while_placeholderNsequential_1_simple_rnn_2_while_less_sequential_1_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: 2&
$sequential_1/simple_rnn_2/while/Less?
(sequential_1/simple_rnn_2/while/IdentityIdentity(sequential_1/simple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: 2*
(sequential_1/simple_rnn_2/while/Identity"]
(sequential_1_simple_rnn_2_while_identity1sequential_1/simple_rnn_2/while/Identity:output:0*A
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
while_cond_20140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_20140___redundant_placeholder03
/while_while_cond_20140___redundant_placeholder13
/while_while_cond_20140___redundant_placeholder23
/while_while_cond_20140___redundant_placeholder3
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
1__inference_simple_rnn_cell_4_layer_call_fn_22218

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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_188412
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
?3
?
while_body_21130
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource??.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_4/MatMul/ReadVariableOp?/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp?
while/simple_rnn_cell_4/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_4/MatMul?
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_4/BiasAdd?
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_4/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_4/MatMul_1?
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/add?
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
?G
?
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_20319

inputs4
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_5/BiasAdd/ReadVariableOp?'simple_rnn_cell_5/MatMul/ReadVariableOp?)simple_rnn_cell_5/MatMul_1/ReadVariableOp?whileD
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
:???????????2
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
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp?
simple_rnn_cell_5/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul?
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp?
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/BiasAdd?
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp?
simple_rnn_cell_5/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul_1?
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/add?
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
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
bodyR
while_body_20253*
condR
while_cond_20252*8
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
:??????????2*
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
:??????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?G
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_19914

inputs4
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_4/BiasAdd/ReadVariableOp?'simple_rnn_cell_4/MatMul/ReadVariableOp?)simple_rnn_cell_4/MatMul_1/ReadVariableOp?whileD
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
:??????????2
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
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp?
simple_rnn_cell_4/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul?
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp?
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/BiasAdd?
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp?
simple_rnn_cell_4/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul_1?
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/add?
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
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
bodyR
while_body_19848*
condR
while_cond_19847*9
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
:???????????*
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
:???????????2
transpose_1?
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_20444
simple_rnn_2_input
simple_rnn_2_20420
simple_rnn_2_20422
simple_rnn_2_20424
simple_rnn_3_20429
simple_rnn_3_20431
simple_rnn_3_20433
dense_1_20438
dense_1_20440
identity??dense_1/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?$simple_rnn_3/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputsimple_rnn_2_20420simple_rnn_2_20422simple_rnn_2_20424*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_200262&
$simple_rnn_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_200612
activation_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_200832
dropout_2/PartitionedCall?
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0simple_rnn_3_20429simple_rnn_3_20431simple_rnn_3_20433*
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_203192&
$simple_rnn_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_203542
activation_3/PartitionedCall?
dropout_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_203762
dropout_3/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_1_20438dense_1_20440*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_204002!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall:` \
,
_output_shapes
:??????????
,
_user_specified_namesimple_rnn_2_input
?
?
,__inference_simple_rnn_2_layer_call_fn_21319

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
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_199142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_simple_rnn_3_layer_call_fn_21845

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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_202072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_20061

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?G
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21308

inputs4
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_4/BiasAdd/ReadVariableOp?'simple_rnn_cell_4/MatMul/ReadVariableOp?)simple_rnn_cell_4/MatMul_1/ReadVariableOp?whileD
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
:??????????2
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
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp?
simple_rnn_cell_4/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul?
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp?
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/BiasAdd?
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp?
simple_rnn_cell_4/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul_1?
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/add?
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
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
bodyR
while_body_21242*
condR
while_cond_21241*9
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
:???????????*
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
:???????????2
transpose_1?
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_19336

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
?
while_body_19727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_5_19749_0#
while_simple_rnn_cell_5_19751_0#
while_simple_rnn_cell_5_19753_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_5_19749!
while_simple_rnn_cell_5_19751!
while_simple_rnn_cell_5_19753??/while/simple_rnn_cell_5/StatefulPartitionedCall?
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
/while/simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_5_19749_0while_simple_rnn_cell_5_19751_0while_simple_rnn_cell_5_19753_0*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_1935321
/while/simple_rnn_cell_5/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_5/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_5/StatefulPartitionedCall:output:10^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_5_19749while_simple_rnn_cell_5_19749_0"@
while_simple_rnn_cell_5_19751while_simple_rnn_cell_5_19751_0"@
while_simple_rnn_cell_5_19753while_simple_rnn_cell_5_19753_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_5/StatefulPartitionedCall/while/simple_rnn_cell_5/StatefulPartitionedCall: 
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
,__inference_sequential_1_layer_call_fn_20493
simple_rnn_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_204742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
,
_output_shapes
:??????????
,
_user_specified_namesimple_rnn_2_input
?
?
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_22190

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

?
B__inference_dense_1_layer_call_and_return_conditional_losses_22147

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

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
?O
?
__inference__traced_save_22408
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_2_simple_rnn_cell_4_kernel_read_readvariableopN
Jsavev2_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_2_simple_rnn_cell_4_bias_read_readvariableopD
@savev2_simple_rnn_3_simple_rnn_cell_5_kernel_read_readvariableopN
Jsavev2_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_3_simple_rnn_cell_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_2_simple_rnn_cell_4_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_2_simple_rnn_cell_4_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_3_simple_rnn_cell_5_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_3_simple_rnn_cell_5_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_2_simple_rnn_cell_4_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_2_simple_rnn_cell_4_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_3_simple_rnn_cell_5_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_3_simple_rnn_cell_5_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_2_simple_rnn_cell_4_kernel_read_readvariableopJsavev2_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_read_readvariableop>savev2_simple_rnn_2_simple_rnn_cell_4_bias_read_readvariableop@savev2_simple_rnn_3_simple_rnn_cell_5_kernel_read_readvariableopJsavev2_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_read_readvariableop>savev2_simple_rnn_3_simple_rnn_cell_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopGsavev2_adam_simple_rnn_2_simple_rnn_cell_4_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_2_simple_rnn_cell_4_bias_m_read_readvariableopGsavev2_adam_simple_rnn_3_simple_rnn_cell_5_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_3_simple_rnn_cell_5_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopGsavev2_adam_simple_rnn_2_simple_rnn_cell_4_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_2_simple_rnn_cell_4_bias_v_read_readvariableopGsavev2_adam_simple_rnn_3_simple_rnn_cell_5_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_3_simple_rnn_cell_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :	2?:?: : : : : :	?:
??:?:	?2:22:2: : : : : : :	2?:?:	?:
??:?:	?2:22:2:	2?:?:	?:
??:?:	?2:22:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	2?:!

_output_shapes	
:?:
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
: :%!

_output_shapes
:	2?:!

_output_shapes	
:?:%!

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
:2:%!

_output_shapes
:	2?:!

_output_shapes	
:?:%!

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
?
?
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_22252

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
`
D__inference_dropout_2_layer_call_and_return_conditional_losses_20078

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
:???????????2
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
:???????????2	
dropoutj
IdentityIdentitydropout:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_22126

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
?Q
?
*sequential_1_simple_rnn_2_while_body_18590P
Lsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_loop_counterV
Rsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_maximum_iterations/
+sequential_1_simple_rnn_2_while_placeholder1
-sequential_1_simple_rnn_2_while_placeholder_11
-sequential_1_simple_rnn_2_while_placeholder_2O
Ksequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_strided_slice_1_0?
?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0V
Rsequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0W
Ssequential_1_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0X
Tsequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0,
(sequential_1_simple_rnn_2_while_identity.
*sequential_1_simple_rnn_2_while_identity_1.
*sequential_1_simple_rnn_2_while_identity_2.
*sequential_1_simple_rnn_2_while_identity_3.
*sequential_1_simple_rnn_2_while_identity_4M
Isequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_strided_slice_1?
?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorT
Psequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resourceU
Qsequential_1_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resourceV
Rsequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource??Hsequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?Gsequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp?Isequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
Qsequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2S
Qsequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Csequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0+sequential_1_simple_rnn_2_while_placeholderZsequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02E
Csequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem?
Gsequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpRsequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02I
Gsequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp?
8sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul	MLCMatMulJsequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul?
Hsequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpSsequential_1_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02J
Hsequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
9sequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAddBiasAddBsequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul:product:0Psequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9sequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd?
Isequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpTsequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02K
Isequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
:sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1	MLCMatMul-sequential_1_simple_rnn_2_while_placeholder_2Qsequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2<
:sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1?
5sequential_1/simple_rnn_2/while/simple_rnn_cell_4/addAddV2Bsequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd:output:0Dsequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????27
5sequential_1/simple_rnn_2/while/simple_rnn_cell_4/add?
6sequential_1/simple_rnn_2/while/simple_rnn_cell_4/TanhTanh9sequential_1/simple_rnn_2/while/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????28
6sequential_1/simple_rnn_2/while/simple_rnn_cell_4/Tanh?
Dsequential_1/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_1_simple_rnn_2_while_placeholder_1+sequential_1_simple_rnn_2_while_placeholder:sequential_1/simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype02F
Dsequential_1/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem?
%sequential_1/simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/simple_rnn_2/while/add/y?
#sequential_1/simple_rnn_2/while/addAddV2+sequential_1_simple_rnn_2_while_placeholder.sequential_1/simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/simple_rnn_2/while/add?
'sequential_1/simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/simple_rnn_2/while/add_1/y?
%sequential_1/simple_rnn_2/while/add_1AddV2Lsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_loop_counter0sequential_1/simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/simple_rnn_2/while/add_1?
(sequential_1/simple_rnn_2/while/IdentityIdentity)sequential_1/simple_rnn_2/while/add_1:z:0I^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2*
(sequential_1/simple_rnn_2/while/Identity?
*sequential_1/simple_rnn_2/while/Identity_1IdentityRsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_maximum_iterationsI^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_1/simple_rnn_2/while/Identity_1?
*sequential_1/simple_rnn_2/while/Identity_2Identity'sequential_1/simple_rnn_2/while/add:z:0I^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_1/simple_rnn_2/while/Identity_2?
*sequential_1/simple_rnn_2/while/Identity_3IdentityTsequential_1/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0I^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_1/simple_rnn_2/while/Identity_3?
*sequential_1/simple_rnn_2/while/Identity_4Identity:sequential_1/simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0I^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2,
*sequential_1/simple_rnn_2/while/Identity_4"]
(sequential_1_simple_rnn_2_while_identity1sequential_1/simple_rnn_2/while/Identity:output:0"a
*sequential_1_simple_rnn_2_while_identity_13sequential_1/simple_rnn_2/while/Identity_1:output:0"a
*sequential_1_simple_rnn_2_while_identity_23sequential_1/simple_rnn_2/while/Identity_2:output:0"a
*sequential_1_simple_rnn_2_while_identity_33sequential_1/simple_rnn_2/while/Identity_3:output:0"a
*sequential_1_simple_rnn_2_while_identity_43sequential_1/simple_rnn_2/while/Identity_4:output:0"?
Isequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_strided_slice_1Ksequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_strided_slice_1_0"?
Qsequential_1_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resourceSsequential_1_simple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"?
Rsequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceTsequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"?
Psequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resourceRsequential_1_simple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"?
?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2?
Hsequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpHsequential_1/simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2?
Gsequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpGsequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp2?
Isequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpIsequential_1/simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
?<
?
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_19673

inputs
simple_rnn_cell_5_19598
simple_rnn_cell_5_19600
simple_rnn_cell_5_19602
identity??)simple_rnn_cell_5/StatefulPartitionedCall?whileD
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
)simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_5_19598simple_rnn_cell_5_19600simple_rnn_cell_5_19602*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_193362+
)simple_rnn_cell_5/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_5_19598simple_rnn_cell_5_19600simple_rnn_cell_5_19602*
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
bodyR
while_body_19610*
condR
while_cond_19609*8
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
IdentityIdentitystrided_slice_3:output:0*^simple_rnn_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2V
)simple_rnn_cell_5/StatefulPartitionedCall)simple_rnn_cell_5/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?3
?
while_body_21376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource??.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_4/MatMul/ReadVariableOp?/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp?
while/simple_rnn_cell_4/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_4/MatMul?
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_4/BiasAdd?
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_4/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_4/MatMul_1?
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/add?
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
while_cond_19097
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_19097___redundant_placeholder03
/while_while_cond_19097___redundant_placeholder13
/while_while_cond_19097___redundant_placeholder23
/while_while_cond_19097___redundant_placeholder3
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
?B
?
simple_rnn_2_while_body_206186
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0J
Fsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0K
Gsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resourceH
Dsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resourceI
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource??;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp?<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02<
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp?
+simple_rnn_2/while/simple_rnn_cell_4/MatMul	MLCMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+simple_rnn_2/while/simple_rnn_cell_4/MatMul?
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02=
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
,simple_rnn_2/while/simple_rnn_cell_4/BiasAddBiasAdd5simple_rnn_2/while/simple_rnn_cell_4/MatMul:product:0Csimple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,simple_rnn_2/while/simple_rnn_cell_4/BiasAdd?
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02>
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
-simple_rnn_2/while/simple_rnn_cell_4/MatMul_1	MLCMatMul simple_rnn_2_while_placeholder_2Dsimple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_rnn_2/while/simple_rnn_cell_4/MatMul_1?
(simple_rnn_2/while/simple_rnn_cell_4/addAddV25simple_rnn_2/while/simple_rnn_cell_4/BiasAdd:output:07simple_rnn_2/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2*
(simple_rnn_2/while/simple_rnn_cell_4/add?
)simple_rnn_2/while/simple_rnn_cell_4/TanhTanh,simple_rnn_2/while/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2+
)simple_rnn_2/while/simple_rnn_cell_4/Tanh?
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1simple_rnn_2_while_placeholder-simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_2/while/add/y?
simple_rnn_2/while/addAddV2simple_rnn_2_while_placeholder!simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/while/addz
simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_2/while/add_1/y?
simple_rnn_2/while/add_1AddV22simple_rnn_2_while_simple_rnn_2_while_loop_counter#simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/while/add_1?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/add_1:z:0<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity?
simple_rnn_2/while/Identity_1Identity8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_1?
simple_rnn_2/while/Identity_2Identitysimple_rnn_2/while/add:z:0<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_2?
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_2/while/Identity_3?
simple_rnn_2/while/Identity_4Identity-simple_rnn_2/while/simple_rnn_cell_4/Tanh:y:0<^simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
simple_rnn_2/while/Identity_4"C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"?
Dsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resourceFsimple_rnn_2_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"?
Esimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"?
Csimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resourceEsimple_rnn_2_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"?
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2z
;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;simple_rnn_2/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2x
:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp:simple_rnn_2/while/simple_rnn_cell_4/MatMul/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
,__inference_sequential_1_layer_call_fn_21084

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
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_205222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?H
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21442
inputs_04
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_4/BiasAdd/ReadVariableOp?'simple_rnn_cell_4/MatMul/ReadVariableOp?)simple_rnn_cell_4/MatMul_1/ReadVariableOp?whileF
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
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp?
simple_rnn_cell_4/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul?
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp?
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/BiasAdd?
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp?
simple_rnn_cell_4/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul_1?
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/add?
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
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
bodyR
while_body_21376*
condR
while_cond_21375*9
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
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_19959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_19959___redundant_placeholder03
/while_while_cond_19959___redundant_placeholder13
/while_while_cond_19959___redundant_placeholder23
/while_while_cond_19959___redundant_placeholder3
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
1__inference_simple_rnn_cell_5_layer_call_fn_22280

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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_193532
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
?	
?
1__inference_simple_rnn_cell_5_layer_call_fn_22266

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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_193362
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
?3
?
while_body_21656
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource??.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_5/MatMul/ReadVariableOp?/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp?
while/simple_rnn_cell_5/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_5/MatMul?
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_5/BiasAdd?
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_5/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_5/MatMul_1?
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/add?
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
?	
 __inference__wrapped_model_18775
simple_rnn_2_inputN
Jsequential_1_simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resourceO
Ksequential_1_simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resourceP
Lsequential_1_simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resourceN
Jsequential_1_simple_rnn_3_simple_rnn_cell_5_matmul_readvariableop_resourceO
Ksequential_1_simple_rnn_3_simple_rnn_cell_5_biasadd_readvariableop_resourceP
Lsequential_1_simple_rnn_3_simple_rnn_cell_5_matmul_1_readvariableop_resource:
6sequential_1_dense_1_mlcmatmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity??+sequential_1/dense_1/BiasAdd/ReadVariableOp?-sequential_1/dense_1/MLCMatMul/ReadVariableOp?Bsequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp?Asequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp?Csequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp?sequential_1/simple_rnn_2/while?Bsequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp?Asequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp?Csequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp?sequential_1/simple_rnn_3/while?
sequential_1/simple_rnn_2/ShapeShapesimple_rnn_2_input*
T0*
_output_shapes
:2!
sequential_1/simple_rnn_2/Shape?
-sequential_1/simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_1/simple_rnn_2/strided_slice/stack?
/sequential_1/simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/simple_rnn_2/strided_slice/stack_1?
/sequential_1/simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/simple_rnn_2/strided_slice/stack_2?
'sequential_1/simple_rnn_2/strided_sliceStridedSlice(sequential_1/simple_rnn_2/Shape:output:06sequential_1/simple_rnn_2/strided_slice/stack:output:08sequential_1/simple_rnn_2/strided_slice/stack_1:output:08sequential_1/simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_1/simple_rnn_2/strided_slice?
%sequential_1/simple_rnn_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2'
%sequential_1/simple_rnn_2/zeros/mul/y?
#sequential_1/simple_rnn_2/zeros/mulMul0sequential_1/simple_rnn_2/strided_slice:output:0.sequential_1/simple_rnn_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/simple_rnn_2/zeros/mul?
&sequential_1/simple_rnn_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_1/simple_rnn_2/zeros/Less/y?
$sequential_1/simple_rnn_2/zeros/LessLess'sequential_1/simple_rnn_2/zeros/mul:z:0/sequential_1/simple_rnn_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_1/simple_rnn_2/zeros/Less?
(sequential_1/simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_1/simple_rnn_2/zeros/packed/1?
&sequential_1/simple_rnn_2/zeros/packedPack0sequential_1/simple_rnn_2/strided_slice:output:01sequential_1/simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_1/simple_rnn_2/zeros/packed?
%sequential_1/simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_1/simple_rnn_2/zeros/Const?
sequential_1/simple_rnn_2/zerosFill/sequential_1/simple_rnn_2/zeros/packed:output:0.sequential_1/simple_rnn_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2!
sequential_1/simple_rnn_2/zeros?
(sequential_1/simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_1/simple_rnn_2/transpose/perm?
#sequential_1/simple_rnn_2/transpose	Transposesimple_rnn_2_input1sequential_1/simple_rnn_2/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2%
#sequential_1/simple_rnn_2/transpose?
!sequential_1/simple_rnn_2/Shape_1Shape'sequential_1/simple_rnn_2/transpose:y:0*
T0*
_output_shapes
:2#
!sequential_1/simple_rnn_2/Shape_1?
/sequential_1/simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_1/simple_rnn_2/strided_slice_1/stack?
1sequential_1/simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_2/strided_slice_1/stack_1?
1sequential_1/simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_2/strided_slice_1/stack_2?
)sequential_1/simple_rnn_2/strided_slice_1StridedSlice*sequential_1/simple_rnn_2/Shape_1:output:08sequential_1/simple_rnn_2/strided_slice_1/stack:output:0:sequential_1/simple_rnn_2/strided_slice_1/stack_1:output:0:sequential_1/simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_1/simple_rnn_2/strided_slice_1?
5sequential_1/simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5sequential_1/simple_rnn_2/TensorArrayV2/element_shape?
'sequential_1/simple_rnn_2/TensorArrayV2TensorListReserve>sequential_1/simple_rnn_2/TensorArrayV2/element_shape:output:02sequential_1/simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_1/simple_rnn_2/TensorArrayV2?
Osequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2Q
Osequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
Asequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_1/simple_rnn_2/transpose:y:0Xsequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor?
/sequential_1/simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_1/simple_rnn_2/strided_slice_2/stack?
1sequential_1/simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_2/strided_slice_2/stack_1?
1sequential_1/simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_2/strided_slice_2/stack_2?
)sequential_1/simple_rnn_2/strided_slice_2StridedSlice'sequential_1/simple_rnn_2/transpose:y:08sequential_1/simple_rnn_2/strided_slice_2/stack:output:0:sequential_1/simple_rnn_2/strided_slice_2/stack_1:output:0:sequential_1/simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)sequential_1/simple_rnn_2/strided_slice_2?
Asequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpJsequential_1_simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02C
Asequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp?
2sequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul	MLCMatMul2sequential_1/simple_rnn_2/strided_slice_2:output:0Isequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2sequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul?
Bsequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpKsequential_1_simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02D
Bsequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
3sequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAddBiasAdd<sequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul:product:0Jsequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????25
3sequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAdd?
Csequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpLsequential_1_simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02E
Csequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
4sequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1	MLCMatMul(sequential_1/simple_rnn_2/zeros:output:0Ksequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????26
4sequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1?
/sequential_1/simple_rnn_2/simple_rnn_cell_4/addAddV2<sequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAdd:output:0>sequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????21
/sequential_1/simple_rnn_2/simple_rnn_cell_4/add?
0sequential_1/simple_rnn_2/simple_rnn_cell_4/TanhTanh3sequential_1/simple_rnn_2/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????22
0sequential_1/simple_rnn_2/simple_rnn_cell_4/Tanh?
7sequential_1/simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7sequential_1/simple_rnn_2/TensorArrayV2_1/element_shape?
)sequential_1/simple_rnn_2/TensorArrayV2_1TensorListReserve@sequential_1/simple_rnn_2/TensorArrayV2_1/element_shape:output:02sequential_1/simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_1/simple_rnn_2/TensorArrayV2_1?
sequential_1/simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential_1/simple_rnn_2/time?
2sequential_1/simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2sequential_1/simple_rnn_2/while/maximum_iterations?
,sequential_1/simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/simple_rnn_2/while/loop_counter?
sequential_1/simple_rnn_2/whileWhile5sequential_1/simple_rnn_2/while/loop_counter:output:0;sequential_1/simple_rnn_2/while/maximum_iterations:output:0'sequential_1/simple_rnn_2/time:output:02sequential_1/simple_rnn_2/TensorArrayV2_1:handle:0(sequential_1/simple_rnn_2/zeros:output:02sequential_1/simple_rnn_2/strided_slice_1:output:0Qsequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_1_simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resourceKsequential_1_simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resourceLsequential_1_simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*6
body.R,
*sequential_1_simple_rnn_2_while_body_18590*6
cond.R,
*sequential_1_simple_rnn_2_while_cond_18589*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2!
sequential_1/simple_rnn_2/while?
Jsequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2L
Jsequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape?
<sequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_1/simple_rnn_2/while:output:3Ssequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02>
<sequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack?
/sequential_1/simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????21
/sequential_1/simple_rnn_2/strided_slice_3/stack?
1sequential_1/simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/simple_rnn_2/strided_slice_3/stack_1?
1sequential_1/simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_2/strided_slice_3/stack_2?
)sequential_1/simple_rnn_2/strided_slice_3StridedSliceEsequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:08sequential_1/simple_rnn_2/strided_slice_3/stack:output:0:sequential_1/simple_rnn_2/strided_slice_3/stack_1:output:0:sequential_1/simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2+
)sequential_1/simple_rnn_2/strided_slice_3?
*sequential_1/simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_1/simple_rnn_2/transpose_1/perm?
%sequential_1/simple_rnn_2/transpose_1	TransposeEsequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:03sequential_1/simple_rnn_2/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2'
%sequential_1/simple_rnn_2/transpose_1?
sequential_1/activation_2/ReluRelu)sequential_1/simple_rnn_2/transpose_1:y:0*
T0*-
_output_shapes
:???????????2 
sequential_1/activation_2/Relu?
sequential_1/dropout_2/IdentityIdentity,sequential_1/activation_2/Relu:activations:0*
T0*-
_output_shapes
:???????????2!
sequential_1/dropout_2/Identity?
sequential_1/simple_rnn_3/ShapeShape(sequential_1/dropout_2/Identity:output:0*
T0*
_output_shapes
:2!
sequential_1/simple_rnn_3/Shape?
-sequential_1/simple_rnn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_1/simple_rnn_3/strided_slice/stack?
/sequential_1/simple_rnn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/simple_rnn_3/strided_slice/stack_1?
/sequential_1/simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_1/simple_rnn_3/strided_slice/stack_2?
'sequential_1/simple_rnn_3/strided_sliceStridedSlice(sequential_1/simple_rnn_3/Shape:output:06sequential_1/simple_rnn_3/strided_slice/stack:output:08sequential_1/simple_rnn_3/strided_slice/stack_1:output:08sequential_1/simple_rnn_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_1/simple_rnn_3/strided_slice?
%sequential_1/simple_rnn_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22'
%sequential_1/simple_rnn_3/zeros/mul/y?
#sequential_1/simple_rnn_3/zeros/mulMul0sequential_1/simple_rnn_3/strided_slice:output:0.sequential_1/simple_rnn_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/simple_rnn_3/zeros/mul?
&sequential_1/simple_rnn_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_1/simple_rnn_3/zeros/Less/y?
$sequential_1/simple_rnn_3/zeros/LessLess'sequential_1/simple_rnn_3/zeros/mul:z:0/sequential_1/simple_rnn_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_1/simple_rnn_3/zeros/Less?
(sequential_1/simple_rnn_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22*
(sequential_1/simple_rnn_3/zeros/packed/1?
&sequential_1/simple_rnn_3/zeros/packedPack0sequential_1/simple_rnn_3/strided_slice:output:01sequential_1/simple_rnn_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_1/simple_rnn_3/zeros/packed?
%sequential_1/simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_1/simple_rnn_3/zeros/Const?
sequential_1/simple_rnn_3/zerosFill/sequential_1/simple_rnn_3/zeros/packed:output:0.sequential_1/simple_rnn_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22!
sequential_1/simple_rnn_3/zeros?
(sequential_1/simple_rnn_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_1/simple_rnn_3/transpose/perm?
#sequential_1/simple_rnn_3/transpose	Transpose(sequential_1/dropout_2/Identity:output:01sequential_1/simple_rnn_3/transpose/perm:output:0*
T0*-
_output_shapes
:???????????2%
#sequential_1/simple_rnn_3/transpose?
!sequential_1/simple_rnn_3/Shape_1Shape'sequential_1/simple_rnn_3/transpose:y:0*
T0*
_output_shapes
:2#
!sequential_1/simple_rnn_3/Shape_1?
/sequential_1/simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_1/simple_rnn_3/strided_slice_1/stack?
1sequential_1/simple_rnn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_3/strided_slice_1/stack_1?
1sequential_1/simple_rnn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_3/strided_slice_1/stack_2?
)sequential_1/simple_rnn_3/strided_slice_1StridedSlice*sequential_1/simple_rnn_3/Shape_1:output:08sequential_1/simple_rnn_3/strided_slice_1/stack:output:0:sequential_1/simple_rnn_3/strided_slice_1/stack_1:output:0:sequential_1/simple_rnn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_1/simple_rnn_3/strided_slice_1?
5sequential_1/simple_rnn_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5sequential_1/simple_rnn_3/TensorArrayV2/element_shape?
'sequential_1/simple_rnn_3/TensorArrayV2TensorListReserve>sequential_1/simple_rnn_3/TensorArrayV2/element_shape:output:02sequential_1/simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_1/simple_rnn_3/TensorArrayV2?
Osequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2Q
Osequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape?
Asequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_1/simple_rnn_3/transpose:y:0Xsequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02C
Asequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor?
/sequential_1/simple_rnn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_1/simple_rnn_3/strided_slice_2/stack?
1sequential_1/simple_rnn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_3/strided_slice_2/stack_1?
1sequential_1/simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_3/strided_slice_2/stack_2?
)sequential_1/simple_rnn_3/strided_slice_2StridedSlice'sequential_1/simple_rnn_3/transpose:y:08sequential_1/simple_rnn_3/strided_slice_2/stack:output:0:sequential_1/simple_rnn_3/strided_slice_2/stack_1:output:0:sequential_1/simple_rnn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2+
)sequential_1/simple_rnn_3/strided_slice_2?
Asequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpJsequential_1_simple_rnn_3_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02C
Asequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp?
2sequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul	MLCMatMul2sequential_1/simple_rnn_3/strided_slice_2:output:0Isequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????224
2sequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul?
Bsequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpKsequential_1_simple_rnn_3_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02D
Bsequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
3sequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAddBiasAdd<sequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul:product:0Jsequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????225
3sequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAdd?
Csequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpLsequential_1_simple_rnn_3_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02E
Csequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
4sequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1	MLCMatMul(sequential_1/simple_rnn_3/zeros:output:0Ksequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????226
4sequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1?
/sequential_1/simple_rnn_3/simple_rnn_cell_5/addAddV2<sequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAdd:output:0>sequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????221
/sequential_1/simple_rnn_3/simple_rnn_cell_5/add?
0sequential_1/simple_rnn_3/simple_rnn_cell_5/TanhTanh3sequential_1/simple_rnn_3/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????222
0sequential_1/simple_rnn_3/simple_rnn_cell_5/Tanh?
7sequential_1/simple_rnn_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7sequential_1/simple_rnn_3/TensorArrayV2_1/element_shape?
)sequential_1/simple_rnn_3/TensorArrayV2_1TensorListReserve@sequential_1/simple_rnn_3/TensorArrayV2_1/element_shape:output:02sequential_1/simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_1/simple_rnn_3/TensorArrayV2_1?
sequential_1/simple_rnn_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential_1/simple_rnn_3/time?
2sequential_1/simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2sequential_1/simple_rnn_3/while/maximum_iterations?
,sequential_1/simple_rnn_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/simple_rnn_3/while/loop_counter?
sequential_1/simple_rnn_3/whileWhile5sequential_1/simple_rnn_3/while/loop_counter:output:0;sequential_1/simple_rnn_3/while/maximum_iterations:output:0'sequential_1/simple_rnn_3/time:output:02sequential_1/simple_rnn_3/TensorArrayV2_1:handle:0(sequential_1/simple_rnn_3/zeros:output:02sequential_1/simple_rnn_3/strided_slice_1:output:0Qsequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_1_simple_rnn_3_simple_rnn_cell_5_matmul_readvariableop_resourceKsequential_1_simple_rnn_3_simple_rnn_cell_5_biasadd_readvariableop_resourceLsequential_1_simple_rnn_3_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*6
body.R,
*sequential_1_simple_rnn_3_while_body_18700*6
cond.R,
*sequential_1_simple_rnn_3_while_cond_18699*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2!
sequential_1/simple_rnn_3/while?
Jsequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2L
Jsequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape?
<sequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_1/simple_rnn_3/while:output:3Ssequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype02>
<sequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack?
/sequential_1/simple_rnn_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????21
/sequential_1/simple_rnn_3/strided_slice_3/stack?
1sequential_1/simple_rnn_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/simple_rnn_3/strided_slice_3/stack_1?
1sequential_1/simple_rnn_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_1/simple_rnn_3/strided_slice_3/stack_2?
)sequential_1/simple_rnn_3/strided_slice_3StridedSliceEsequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:08sequential_1/simple_rnn_3/strided_slice_3/stack:output:0:sequential_1/simple_rnn_3/strided_slice_3/stack_1:output:0:sequential_1/simple_rnn_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2+
)sequential_1/simple_rnn_3/strided_slice_3?
*sequential_1/simple_rnn_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential_1/simple_rnn_3/transpose_1/perm?
%sequential_1/simple_rnn_3/transpose_1	TransposeEsequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:03sequential_1/simple_rnn_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22'
%sequential_1/simple_rnn_3/transpose_1?
sequential_1/activation_3/ReluRelu2sequential_1/simple_rnn_3/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22 
sequential_1/activation_3/Relu?
sequential_1/dropout_3/IdentityIdentity,sequential_1/activation_3/Relu:activations:0*
T0*'
_output_shapes
:?????????22!
sequential_1/dropout_3/Identity?
-sequential_1/dense_1/MLCMatMul/ReadVariableOpReadVariableOp6sequential_1_dense_1_mlcmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02/
-sequential_1/dense_1/MLCMatMul/ReadVariableOp?
sequential_1/dense_1/MLCMatMul	MLCMatMul(sequential_1/dropout_3/Identity:output:05sequential_1/dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_1/dense_1/MLCMatMul?
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp?
sequential_1/dense_1/BiasAddBiasAdd(sequential_1/dense_1/MLCMatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_1/BiasAdd?
sequential_1/dense_1/TanhTanh%sequential_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_1/Tanh?
IdentityIdentitysequential_1/dense_1/Tanh:y:0,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/MLCMatMul/ReadVariableOpC^sequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpB^sequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpD^sequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp ^sequential_1/simple_rnn_2/whileC^sequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOpB^sequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOpD^sequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp ^sequential_1/simple_rnn_3/while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/MLCMatMul/ReadVariableOp-sequential_1/dense_1/MLCMatMul/ReadVariableOp2?
Bsequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpBsequential_1/simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp2?
Asequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpAsequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp2?
Csequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpCsequential_1/simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp2B
sequential_1/simple_rnn_2/whilesequential_1/simple_rnn_2/while2?
Bsequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOpBsequential_1/simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp2?
Asequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOpAsequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp2?
Csequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOpCsequential_1/simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp2B
sequential_1/simple_rnn_3/whilesequential_1/simple_rnn_3/while:` \
,
_output_shapes
:??????????
,
_user_specified_namesimple_rnn_2_input
?
?
while_cond_21655
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21655___redundant_placeholder03
/while_while_cond_21655___redundant_placeholder13
/while_while_cond_21655___redundant_placeholder23
/while_while_cond_21655___redundant_placeholder3
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
while_cond_19726
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_19726___redundant_placeholder03
/while_while_cond_19726___redundant_placeholder13
/while_while_cond_19726___redundant_placeholder23
/while_while_cond_19726___redundant_placeholder3
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
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_20083

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?G
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21196

inputs4
0simple_rnn_cell_4_matmul_readvariableop_resource5
1simple_rnn_cell_4_biasadd_readvariableop_resource6
2simple_rnn_cell_4_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_4/BiasAdd/ReadVariableOp?'simple_rnn_cell_4/MatMul/ReadVariableOp?)simple_rnn_cell_4/MatMul_1/ReadVariableOp?whileD
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
:??????????2
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
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'simple_rnn_cell_4/MatMul/ReadVariableOp?
simple_rnn_cell_4/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul?
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(simple_rnn_cell_4/BiasAdd/ReadVariableOp?
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/BiasAdd?
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)simple_rnn_cell_4/MatMul_1/ReadVariableOp?
simple_rnn_cell_4/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/MatMul_1?
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/add?
simple_rnn_cell_4/TanhTanhsimple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
simple_rnn_cell_4/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
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
bodyR
while_body_21130*
condR
while_cond_21129*9
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
:???????????*
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
:???????????2
transpose_1?
IdentityIdentitytranspose_1:y:0)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?#
?
while_body_19215
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_4_19237_0#
while_simple_rnn_cell_4_19239_0#
while_simple_rnn_cell_4_19241_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_4_19237!
while_simple_rnn_cell_4_19239!
while_simple_rnn_cell_4_19241??/while/simple_rnn_cell_4/StatefulPartitionedCall?
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
/while/simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_4_19237_0while_simple_rnn_cell_4_19239_0while_simple_rnn_cell_4_19241_0*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1884121
/while/simple_rnn_cell_4/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_4/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_4/StatefulPartitionedCall:output:10^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_4_19237while_simple_rnn_cell_4_19237_0"@
while_simple_rnn_cell_4_19239while_simple_rnn_cell_4_19239_0"@
while_simple_rnn_cell_4_19241while_simple_rnn_cell_4_19241_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_4/StatefulPartitionedCall/while/simple_rnn_cell_4/StatefulPartitionedCall: 
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
?B
?
simple_rnn_3_while_body_209676
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_25
1simple_rnn_3_while_simple_rnn_3_strided_slice_1_0q
msimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resource_0J
Fsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0K
Gsimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
simple_rnn_3_while_identity!
simple_rnn_3_while_identity_1!
simple_rnn_3_while_identity_2!
simple_rnn_3_while_identity_3!
simple_rnn_3_while_identity_43
/simple_rnn_3_while_simple_rnn_3_strided_slice_1o
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resourceH
Dsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resourceI
Esimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource??;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp?<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2F
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_3_while_placeholderMsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype028
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02<
:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp?
+simple_rnn_3/while/simple_rnn_cell_5/MatMul	MLCMatMul=simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22-
+simple_rnn_3/while/simple_rnn_cell_5/MatMul?
;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02=
;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
,simple_rnn_3/while/simple_rnn_cell_5/BiasAddBiasAdd5simple_rnn_3/while/simple_rnn_cell_5/MatMul:product:0Csimple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22.
,simple_rnn_3/while/simple_rnn_cell_5/BiasAdd?
<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02>
<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
-simple_rnn_3/while/simple_rnn_cell_5/MatMul_1	MLCMatMul simple_rnn_3_while_placeholder_2Dsimple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22/
-simple_rnn_3/while/simple_rnn_cell_5/MatMul_1?
(simple_rnn_3/while/simple_rnn_cell_5/addAddV25simple_rnn_3/while/simple_rnn_cell_5/BiasAdd:output:07simple_rnn_3/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22*
(simple_rnn_3/while/simple_rnn_cell_5/add?
)simple_rnn_3/while/simple_rnn_cell_5/TanhTanh,simple_rnn_3/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22+
)simple_rnn_3/while/simple_rnn_cell_5/Tanh?
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_3_while_placeholder_1simple_rnn_3_while_placeholder-simple_rnn_3/while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_3/while/add/y?
simple_rnn_3/while/addAddV2simple_rnn_3_while_placeholder!simple_rnn_3/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/while/addz
simple_rnn_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_3/while/add_1/y?
simple_rnn_3/while/add_1AddV22simple_rnn_3_while_simple_rnn_3_while_loop_counter#simple_rnn_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/while/add_1?
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/add_1:z:0<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity?
simple_rnn_3/while/Identity_1Identity8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_1?
simple_rnn_3/while/Identity_2Identitysimple_rnn_3/while/add:z:0<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_2?
simple_rnn_3/while/Identity_3IdentityGsimple_rnn_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_3?
simple_rnn_3/while/Identity_4Identity-simple_rnn_3/while/simple_rnn_cell_5/Tanh:y:0<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
simple_rnn_3/while/Identity_4"C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0"G
simple_rnn_3_while_identity_1&simple_rnn_3/while/Identity_1:output:0"G
simple_rnn_3_while_identity_2&simple_rnn_3/while/Identity_2:output:0"G
simple_rnn_3_while_identity_3&simple_rnn_3/while/Identity_3:output:0"G
simple_rnn_3_while_identity_4&simple_rnn_3/while/Identity_4:output:0"d
/simple_rnn_3_while_simple_rnn_3_strided_slice_11simple_rnn_3_while_simple_rnn_3_strided_slice_1_0"?
Dsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resourceFsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"?
Esimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceGsimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"?
Csimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resourceEsimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"?
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensormsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2z
;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2x
:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp2|
<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
while_cond_22013
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22013___redundant_placeholder03
/while_while_cond_22013___redundant_placeholder13
/while_while_cond_22013___redundant_placeholder23
/while_while_cond_22013___redundant_placeholder3
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
?3
?
while_body_19960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource??.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_4/MatMul/ReadVariableOp?/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp?
while/simple_rnn_cell_4/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_4/MatMul?
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_4/BiasAdd?
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_4/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_4/MatMul_1?
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/add?
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
?
,__inference_simple_rnn_2_layer_call_fn_21576
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_192782
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
?3
?
while_body_19848
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource??.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_4/MatMul/ReadVariableOp?/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp?
while/simple_rnn_cell_4/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_4/MatMul?
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_4/BiasAdd?
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_4/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_4/MatMul_1?
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/add?
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
?
*sequential_1_simple_rnn_3_while_cond_18699P
Lsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_loop_counterV
Rsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_maximum_iterations/
+sequential_1_simple_rnn_3_while_placeholder1
-sequential_1_simple_rnn_3_while_placeholder_11
-sequential_1_simple_rnn_3_while_placeholder_2R
Nsequential_1_simple_rnn_3_while_less_sequential_1_simple_rnn_3_strided_slice_1g
csequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_cond_18699___redundant_placeholder0g
csequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_cond_18699___redundant_placeholder1g
csequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_cond_18699___redundant_placeholder2g
csequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_cond_18699___redundant_placeholder3,
(sequential_1_simple_rnn_3_while_identity
?
$sequential_1/simple_rnn_3/while/LessLess+sequential_1_simple_rnn_3_while_placeholderNsequential_1_simple_rnn_3_while_less_sequential_1_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: 2&
$sequential_1/simple_rnn_3/while/Less?
(sequential_1/simple_rnn_3/while/IdentityIdentity(sequential_1/simple_rnn_3/while/Less:z:0*
T0
*
_output_shapes
: 2*
(sequential_1/simple_rnn_3/while/Identity"]
(sequential_1_simple_rnn_3_while_identity1sequential_1/simple_rnn_3/while/Identity:output:0*@
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
c
G__inference_activation_3_layer_call_and_return_conditional_losses_20354

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
?G
?
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_20207

inputs4
0simple_rnn_cell_5_matmul_readvariableop_resource5
1simple_rnn_cell_5_biasadd_readvariableop_resource6
2simple_rnn_cell_5_matmul_1_readvariableop_resource
identity??(simple_rnn_cell_5/BiasAdd/ReadVariableOp?'simple_rnn_cell_5/MatMul/ReadVariableOp?)simple_rnn_cell_5/MatMul_1/ReadVariableOp?whileD
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
:???????????2
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
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02)
'simple_rnn_cell_5/MatMul/ReadVariableOp?
simple_rnn_cell_5/MatMul	MLCMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul?
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(simple_rnn_cell_5/BiasAdd/ReadVariableOp?
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/BiasAdd?
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype02+
)simple_rnn_cell_5/MatMul_1/ReadVariableOp?
simple_rnn_cell_5/MatMul_1	MLCMatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/MatMul_1?
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/add?
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
simple_rnn_cell_5/Tanh?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
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
bodyR
while_body_20141*
condR
while_cond_20140*8
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
:??????????2*
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
:??????????22
transpose_1?
IdentityIdentitystrided_slice_3:output:0)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????:::2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?<
?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_19278

inputs
simple_rnn_cell_4_19203
simple_rnn_cell_4_19205
simple_rnn_cell_4_19207
identity??)simple_rnn_cell_4/StatefulPartitionedCall?whileD
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
)simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_4_19203simple_rnn_cell_4_19205simple_rnn_cell_4_19207*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_188412+
)simple_rnn_cell_4/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_4_19203simple_rnn_cell_4_19205simple_rnn_cell_4_19207*
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
bodyR
while_body_19215*
condR
while_cond_19214*9
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
IdentityIdentitytranspose_1:y:0*^simple_rnn_cell_4/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2V
)simple_rnn_cell_4/StatefulPartitionedCall)simple_rnn_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_20400

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

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
?3
?
while_body_21902
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource??.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_5/MatMul/ReadVariableOp?/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp?
while/simple_rnn_cell_5/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_5/MatMul?
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_5/BiasAdd?
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_5/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_5/MatMul_1?
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/add?
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
`
D__inference_dropout_3_layer_call_and_return_conditional_losses_20371

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
?

?
simple_rnn_3_while_cond_209666
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_28
4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1M
Isimple_rnn_3_while_simple_rnn_3_while_cond_20966___redundant_placeholder0M
Isimple_rnn_3_while_simple_rnn_3_while_cond_20966___redundant_placeholder1M
Isimple_rnn_3_while_simple_rnn_3_while_cond_20966___redundant_placeholder2M
Isimple_rnn_3_while_simple_rnn_3_while_cond_20966___redundant_placeholder3
simple_rnn_3_while_identity
?
simple_rnn_3/while/LessLesssimple_rnn_3_while_placeholder4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_3/while/Less?
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_3/while/Identity"C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0*@
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
?3
?
while_body_20141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource??.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_5/MatMul/ReadVariableOp?/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp?
while/simple_rnn_cell_5/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_5/MatMul?
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_5/BiasAdd?
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_5/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_5/MatMul_1?
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/add?
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_20811

inputsA
=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resourceB
>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resourceC
?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resourceA
=simple_rnn_3_simple_rnn_cell_5_matmul_readvariableop_resourceB
>simple_rnn_3_simple_rnn_cell_5_biasadd_readvariableop_resourceC
?simple_rnn_3_simple_rnn_cell_5_matmul_1_readvariableop_resource-
)dense_1_mlcmatmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp?4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp?6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp?simple_rnn_2/while?5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp?4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp?6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp?simple_rnn_3/while^
simple_rnn_2/ShapeShapeinputs*
T0*
_output_shapes
:2
simple_rnn_2/Shape?
 simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_2/strided_slice/stack?
"simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_2/strided_slice/stack_1?
"simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_2/strided_slice/stack_2?
simple_rnn_2/strided_sliceStridedSlicesimple_rnn_2/Shape:output:0)simple_rnn_2/strided_slice/stack:output:0+simple_rnn_2/strided_slice/stack_1:output:0+simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_2/strided_slicew
simple_rnn_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/mul/y?
simple_rnn_2/zeros/mulMul#simple_rnn_2/strided_slice:output:0!simple_rnn_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/zeros/muly
simple_rnn_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/Less/y?
simple_rnn_2/zeros/LessLesssimple_rnn_2/zeros/mul:z:0"simple_rnn_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_2/zeros/Less}
simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_2/zeros/packed/1?
simple_rnn_2/zeros/packedPack#simple_rnn_2/strided_slice:output:0$simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_2/zeros/packedy
simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_2/zeros/Const?
simple_rnn_2/zerosFill"simple_rnn_2/zeros/packed:output:0!simple_rnn_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
simple_rnn_2/zeros?
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_2/transpose/perm?
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
simple_rnn_2/transposev
simple_rnn_2/Shape_1Shapesimple_rnn_2/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_2/Shape_1?
"simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_2/strided_slice_1/stack?
$simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_1/stack_1?
$simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_1/stack_2?
simple_rnn_2/strided_slice_1StridedSlicesimple_rnn_2/Shape_1:output:0+simple_rnn_2/strided_slice_1/stack:output:0-simple_rnn_2/strided_slice_1/stack_1:output:0-simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_2/strided_slice_1?
(simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_2/TensorArrayV2/element_shape?
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_2/TensorArrayV2?
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_2/strided_slice_2/stack?
$simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_2/stack_1?
$simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_2/stack_2?
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
simple_rnn_2/strided_slice_2?
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp?
%simple_rnn_2/simple_rnn_cell_4/MatMul	MLCMatMul%simple_rnn_2/strided_slice_2:output:0<simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%simple_rnn_2/simple_rnn_cell_4/MatMul?
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
&simple_rnn_2/simple_rnn_cell_4/BiasAddBiasAdd/simple_rnn_2/simple_rnn_cell_4/MatMul:product:0=simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&simple_rnn_2/simple_rnn_cell_4/BiasAdd?
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
'simple_rnn_2/simple_rnn_cell_4/MatMul_1	MLCMatMulsimple_rnn_2/zeros:output:0>simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'simple_rnn_2/simple_rnn_cell_4/MatMul_1?
"simple_rnn_2/simple_rnn_cell_4/addAddV2/simple_rnn_2/simple_rnn_cell_4/BiasAdd:output:01simple_rnn_2/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2$
"simple_rnn_2/simple_rnn_cell_4/add?
#simple_rnn_2/simple_rnn_cell_4/TanhTanh&simple_rnn_2/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2%
#simple_rnn_2/simple_rnn_cell_4/Tanh?
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2,
*simple_rnn_2/TensorArrayV2_1/element_shape?
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_2/TensorArrayV2_1h
simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_2/time?
%simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_2/while/maximum_iterations?
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_2/while/loop_counter?
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_2_simple_rnn_cell_4_matmul_readvariableop_resource>simple_rnn_2_simple_rnn_cell_4_biasadd_readvariableop_resource?simple_rnn_2_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_2_while_body_20618*)
cond!R
simple_rnn_2_while_cond_20617*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
simple_rnn_2/while?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype021
/simple_rnn_2/TensorArrayV2Stack/TensorListStack?
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_2/strided_slice_3/stack?
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_2/strided_slice_3/stack_1?
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_2/strided_slice_3/stack_2?
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_2/strided_slice_3?
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_2/transpose_1/perm?
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
simple_rnn_2/transpose_1?
activation_2/ReluRelusimple_rnn_2/transpose_1:y:0*
T0*-
_output_shapes
:???????????2
activation_2/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulactivation_2/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*-
_output_shapes
:???????????2
dropout_2/dropout/Mulu
dropout_2/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_2/dropout/rater
dropout_2/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_2/dropout/seed?
dropout_2/dropout
MLCDropoutactivation_2/Relu:activations:0dropout_2/dropout/rate:output:0dropout_2/dropout/seed:output:0*
T0*-
_output_shapes
:???????????2
dropout_2/dropoutr
simple_rnn_3/ShapeShapedropout_2/dropout:output:0*
T0*
_output_shapes
:2
simple_rnn_3/Shape?
 simple_rnn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_3/strided_slice/stack?
"simple_rnn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_3/strided_slice/stack_1?
"simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_3/strided_slice/stack_2?
simple_rnn_3/strided_sliceStridedSlicesimple_rnn_3/Shape:output:0)simple_rnn_3/strided_slice/stack:output:0+simple_rnn_3/strided_slice/stack_1:output:0+simple_rnn_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_3/strided_slicev
simple_rnn_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_3/zeros/mul/y?
simple_rnn_3/zeros/mulMul#simple_rnn_3/strided_slice:output:0!simple_rnn_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/zeros/muly
simple_rnn_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
simple_rnn_3/zeros/Less/y?
simple_rnn_3/zeros/LessLesssimple_rnn_3/zeros/mul:z:0"simple_rnn_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/zeros/Less|
simple_rnn_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
simple_rnn_3/zeros/packed/1?
simple_rnn_3/zeros/packedPack#simple_rnn_3/strided_slice:output:0$simple_rnn_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
simple_rnn_3/zeros/packedy
simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
simple_rnn_3/zeros/Const?
simple_rnn_3/zerosFill"simple_rnn_3/zeros/packed:output:0!simple_rnn_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
simple_rnn_3/zeros?
simple_rnn_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_3/transpose/perm?
simple_rnn_3/transpose	Transposedropout_2/dropout:output:0$simple_rnn_3/transpose/perm:output:0*
T0*-
_output_shapes
:???????????2
simple_rnn_3/transposev
simple_rnn_3/Shape_1Shapesimple_rnn_3/transpose:y:0*
T0*
_output_shapes
:2
simple_rnn_3/Shape_1?
"simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_3/strided_slice_1/stack?
$simple_rnn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_1/stack_1?
$simple_rnn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_1/stack_2?
simple_rnn_3/strided_slice_1StridedSlicesimple_rnn_3/Shape_1:output:0+simple_rnn_3/strided_slice_1/stack:output:0-simple_rnn_3/strided_slice_1/stack_1:output:0-simple_rnn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_3/strided_slice_1?
(simple_rnn_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(simple_rnn_3/TensorArrayV2/element_shape?
simple_rnn_3/TensorArrayV2TensorListReserve1simple_rnn_3/TensorArrayV2/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_3/TensorArrayV2?
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2D
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape?
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_3/transpose:y:0Ksimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensor?
"simple_rnn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_3/strided_slice_2/stack?
$simple_rnn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_2/stack_1?
$simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_2/stack_2?
simple_rnn_3/strided_slice_2StridedSlicesimple_rnn_3/transpose:y:0+simple_rnn_3/strided_slice_2/stack:output:0-simple_rnn_3/strided_slice_2/stack_1:output:0-simple_rnn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
simple_rnn_3/strided_slice_2?
4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp=simple_rnn_3_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype026
4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp?
%simple_rnn_3/simple_rnn_cell_5/MatMul	MLCMatMul%simple_rnn_3/strided_slice_2:output:0<simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22'
%simple_rnn_3/simple_rnn_cell_5/MatMul?
5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_3_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype027
5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
&simple_rnn_3/simple_rnn_cell_5/BiasAddBiasAdd/simple_rnn_3/simple_rnn_cell_5/MatMul:product:0=simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22(
&simple_rnn_3/simple_rnn_cell_5/BiasAdd?
6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_3_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:22*
dtype028
6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
'simple_rnn_3/simple_rnn_cell_5/MatMul_1	MLCMatMulsimple_rnn_3/zeros:output:0>simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22)
'simple_rnn_3/simple_rnn_cell_5/MatMul_1?
"simple_rnn_3/simple_rnn_cell_5/addAddV2/simple_rnn_3/simple_rnn_cell_5/BiasAdd:output:01simple_rnn_3/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22$
"simple_rnn_3/simple_rnn_cell_5/add?
#simple_rnn_3/simple_rnn_cell_5/TanhTanh&simple_rnn_3/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22%
#simple_rnn_3/simple_rnn_cell_5/Tanh?
*simple_rnn_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2,
*simple_rnn_3/TensorArrayV2_1/element_shape?
simple_rnn_3/TensorArrayV2_1TensorListReserve3simple_rnn_3/TensorArrayV2_1/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_3/TensorArrayV2_1h
simple_rnn_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_3/time?
%simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%simple_rnn_3/while/maximum_iterations?
simple_rnn_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_3/while/loop_counter?
simple_rnn_3/whileWhile(simple_rnn_3/while/loop_counter:output:0.simple_rnn_3/while/maximum_iterations:output:0simple_rnn_3/time:output:0%simple_rnn_3/TensorArrayV2_1:handle:0simple_rnn_3/zeros:output:0%simple_rnn_3/strided_slice_1:output:0Dsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_3_simple_rnn_cell_5_matmul_readvariableop_resource>simple_rnn_3_simple_rnn_cell_5_biasadd_readvariableop_resource?simple_rnn_3_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_3_while_body_20732*)
cond!R
simple_rnn_3_while_cond_20731*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
simple_rnn_3/while?
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2?
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape?
/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_3/while:output:3Fsimple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????2*
element_dtype021
/simple_rnn_3/TensorArrayV2Stack/TensorListStack?
"simple_rnn_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"simple_rnn_3/strided_slice_3/stack?
$simple_rnn_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_3/strided_slice_3/stack_1?
$simple_rnn_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_3/strided_slice_3/stack_2?
simple_rnn_3/strided_slice_3StridedSlice8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_3/strided_slice_3/stack:output:0-simple_rnn_3/strided_slice_3/stack_1:output:0-simple_rnn_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
simple_rnn_3/strided_slice_3?
simple_rnn_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_3/transpose_1/perm?
simple_rnn_3/transpose_1	Transpose8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????22
simple_rnn_3/transpose_1?
activation_3/ReluRelu%simple_rnn_3/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????22
activation_3/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMulactivation_3/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_3/dropout/Mulu
dropout_3/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_3/dropout/rater
dropout_3/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout_3/dropout/seed?
dropout_3/dropout
MLCDropoutactivation_3/Relu:activations:0dropout_3/dropout/rate:output:0dropout_3/dropout/seed:output:0*
T0*'
_output_shapes
:?????????22
dropout_3/dropout?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMuldropout_3/dropout:output:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Tanh?
IdentityIdentitydense_1/Tanh:y:0^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp6^simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp5^simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp^simple_rnn_2/while6^simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp5^simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp7^simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp^simple_rnn_3/while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp2n
5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp5simple_rnn_2/simple_rnn_cell_4/BiasAdd/ReadVariableOp2l
4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp4simple_rnn_2/simple_rnn_cell_4/MatMul/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp6simple_rnn_2/simple_rnn_cell_4/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while2n
5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp5simple_rnn_3/simple_rnn_cell_5/BiasAdd/ReadVariableOp2l
4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp4simple_rnn_3/simple_rnn_cell_5/MatMul/ReadVariableOp2p
6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp6simple_rnn_3/simple_rnn_cell_5/MatMul_1/ReadVariableOp2(
simple_rnn_3/whilesimple_rnn_3/while:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_21581

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_activation_2_layer_call_fn_21586

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
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_200612
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_dropout_3_layer_call_and_return_conditional_losses_22121

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
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_20417
simple_rnn_2_input
simple_rnn_2_20049
simple_rnn_2_20051
simple_rnn_2_20053
simple_rnn_3_20342
simple_rnn_3_20344
simple_rnn_3_20346
dense_1_20411
dense_1_20413
identity??dense_1/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?$simple_rnn_3/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputsimple_rnn_2_20049simple_rnn_2_20051simple_rnn_2_20053*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_199142&
$simple_rnn_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_200612
activation_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_200782
dropout_2/PartitionedCall?
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0simple_rnn_3_20342simple_rnn_3_20344simple_rnn_3_20346*
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_202072&
$simple_rnn_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_203542
activation_3/PartitionedCall?
dropout_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_203712
dropout_3/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_1_20411dense_1_20413*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_204002!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall:` \
,
_output_shapes
:??????????
,
_user_specified_namesimple_rnn_2_input
?B
?
simple_rnn_3_while_body_207326
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_25
1simple_rnn_3_while_simple_rnn_3_strided_slice_1_0q
msimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resource_0J
Fsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0K
Gsimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
simple_rnn_3_while_identity!
simple_rnn_3_while_identity_1!
simple_rnn_3_while_identity_2!
simple_rnn_3_while_identity_3!
simple_rnn_3_while_identity_43
/simple_rnn_3_while_simple_rnn_3_strided_slice_1o
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resourceH
Dsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resourceI
Esimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource??;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp?<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2F
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_3_while_placeholderMsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype028
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem?
:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02<
:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp?
+simple_rnn_3/while/simple_rnn_cell_5/MatMul	MLCMatMul=simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22-
+simple_rnn_3/while/simple_rnn_cell_5/MatMul?
;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype02=
;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
,simple_rnn_3/while/simple_rnn_cell_5/BiasAddBiasAdd5simple_rnn_3/while/simple_rnn_cell_5/MatMul:product:0Csimple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22.
,simple_rnn_3/while/simple_rnn_cell_5/BiasAdd?
<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype02>
<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
-simple_rnn_3/while/simple_rnn_cell_5/MatMul_1	MLCMatMul simple_rnn_3_while_placeholder_2Dsimple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22/
-simple_rnn_3/while/simple_rnn_cell_5/MatMul_1?
(simple_rnn_3/while/simple_rnn_cell_5/addAddV25simple_rnn_3/while/simple_rnn_cell_5/BiasAdd:output:07simple_rnn_3/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22*
(simple_rnn_3/while/simple_rnn_cell_5/add?
)simple_rnn_3/while/simple_rnn_cell_5/TanhTanh,simple_rnn_3/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22+
)simple_rnn_3/while/simple_rnn_cell_5/Tanh?
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_3_while_placeholder_1simple_rnn_3_while_placeholder-simple_rnn_3/while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_3/while/add/y?
simple_rnn_3/while/addAddV2simple_rnn_3_while_placeholder!simple_rnn_3/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/while/addz
simple_rnn_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_3/while/add_1/y?
simple_rnn_3/while/add_1AddV22simple_rnn_3_while_simple_rnn_3_while_loop_counter#simple_rnn_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_3/while/add_1?
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/add_1:z:0<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity?
simple_rnn_3/while/Identity_1Identity8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_1?
simple_rnn_3/while/Identity_2Identitysimple_rnn_3/while/add:z:0<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_2?
simple_rnn_3/while/Identity_3IdentityGsimple_rnn_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
simple_rnn_3/while/Identity_3?
simple_rnn_3/while/Identity_4Identity-simple_rnn_3/while/simple_rnn_cell_5/Tanh:y:0<^simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
simple_rnn_3/while/Identity_4"C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0"G
simple_rnn_3_while_identity_1&simple_rnn_3/while/Identity_1:output:0"G
simple_rnn_3_while_identity_2&simple_rnn_3/while/Identity_2:output:0"G
simple_rnn_3_while_identity_3&simple_rnn_3/while/Identity_3:output:0"G
simple_rnn_3_while_identity_4&simple_rnn_3/while/Identity_4:output:0"d
/simple_rnn_3_while_simple_rnn_3_strided_slice_11simple_rnn_3_while_simple_rnn_3_strided_slice_1_0"?
Dsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resourceFsimple_rnn_3_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"?
Esimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceGsimple_rnn_3_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"?
Csimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resourceEsimple_rnn_3_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"?
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensormsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2z
;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;simple_rnn_3/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2x
:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp:simple_rnn_3/while/simple_rnn_cell_5/MatMul/ReadVariableOp2|
<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp<simple_rnn_3/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_20376

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
?
E
)__inference_dropout_3_layer_call_fn_22131

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
GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_203712
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
?
|
'__inference_dense_1_layer_call_fn_22156

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
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_204002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_22523
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate<
8assignvariableop_7_simple_rnn_2_simple_rnn_cell_4_kernelF
Bassignvariableop_8_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel:
6assignvariableop_9_simple_rnn_2_simple_rnn_cell_4_bias=
9assignvariableop_10_simple_rnn_3_simple_rnn_cell_5_kernelG
Cassignvariableop_11_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel;
7assignvariableop_12_simple_rnn_3_simple_rnn_cell_5_bias
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1
assignvariableop_17_total_2
assignvariableop_18_count_2-
)assignvariableop_19_adam_dense_1_kernel_m+
'assignvariableop_20_adam_dense_1_bias_mD
@assignvariableop_21_adam_simple_rnn_2_simple_rnn_cell_4_kernel_mN
Jassignvariableop_22_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_mB
>assignvariableop_23_adam_simple_rnn_2_simple_rnn_cell_4_bias_mD
@assignvariableop_24_adam_simple_rnn_3_simple_rnn_cell_5_kernel_mN
Jassignvariableop_25_adam_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_mB
>assignvariableop_26_adam_simple_rnn_3_simple_rnn_cell_5_bias_m-
)assignvariableop_27_adam_dense_1_kernel_v+
'assignvariableop_28_adam_dense_1_bias_vD
@assignvariableop_29_adam_simple_rnn_2_simple_rnn_cell_4_kernel_vN
Jassignvariableop_30_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_vB
>assignvariableop_31_adam_simple_rnn_2_simple_rnn_cell_4_bias_vD
@assignvariableop_32_adam_simple_rnn_3_simple_rnn_cell_5_kernel_vN
Jassignvariableop_33_adam_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_vB
>assignvariableop_34_adam_simple_rnn_3_simple_rnn_cell_5_bias_v
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
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp8assignvariableop_7_simple_rnn_2_simple_rnn_cell_4_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpBassignvariableop_8_simple_rnn_2_simple_rnn_cell_4_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_simple_rnn_2_simple_rnn_cell_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_simple_rnn_3_simple_rnn_cell_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpCassignvariableop_11_simple_rnn_3_simple_rnn_cell_5_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_simple_rnn_3_simple_rnn_cell_5_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_simple_rnn_2_simple_rnn_cell_4_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpJassignvariableop_22_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_simple_rnn_2_simple_rnn_cell_4_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp@assignvariableop_24_adam_simple_rnn_3_simple_rnn_cell_5_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpJassignvariableop_25_adam_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_simple_rnn_3_simple_rnn_cell_5_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_simple_rnn_2_simple_rnn_cell_4_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpJassignvariableop_30_adam_simple_rnn_2_simple_rnn_cell_4_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_simple_rnn_2_simple_rnn_cell_4_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_simple_rnn_3_simple_rnn_cell_5_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpJassignvariableop_33_adam_simple_rnn_3_simple_rnn_cell_5_recurrent_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adam_simple_rnn_3_simple_rnn_cell_5_bias_vIdentity_34:output:0"/device:CPU:0*
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
?3
?
while_body_20253
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource??.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_5/MatMul/ReadVariableOp?/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp?
while/simple_rnn_cell_5/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_5/MatMul?
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_5/BiasAdd?
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_5/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_5/MatMul_1?
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/add?
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
?3
?
while_body_21768
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_5_matmul_readvariableop_resource;
7while_simple_rnn_cell_5_biasadd_readvariableop_resource<
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource??.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_5/MatMul/ReadVariableOp?/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?2*
dtype02/
-while/simple_rnn_cell_5/MatMul/ReadVariableOp?
while/simple_rnn_cell_5/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22 
while/simple_rnn_cell_5/MatMul?
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:2*
dtype020
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22!
while/simple_rnn_cell_5/BiasAdd?
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:22*
dtype021
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_5/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 while/simple_rnn_cell_5/MatMul_1?
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/add?
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:?????????22
while/simple_rnn_cell_5/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
while_cond_21487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21487___redundant_placeholder03
/while_while_cond_21487___redundant_placeholder13
/while_while_cond_21487___redundant_placeholder23
/while_while_cond_21487___redundant_placeholder3
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
?3
?
while_body_21488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_4_matmul_readvariableop_resource;
7while_simple_rnn_cell_4_biasadd_readvariableop_resource<
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource??.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_4/MatMul/ReadVariableOp?/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
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
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02/
-while/simple_rnn_cell_4/MatMul/ReadVariableOp?
while/simple_rnn_cell_4/MatMul	MLCMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
while/simple_rnn_cell_4/MatMul?
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype020
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/simple_rnn_cell_4/BiasAdd?
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_4/MatMul_1	MLCMatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/simple_rnn_cell_4/MatMul_1?
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/add?
while/simple_rnn_cell_4/TanhTanhwhile/simple_rnn_cell_4/add:z:0*
T0*(
_output_shapes
:??????????2
while/simple_rnn_cell_4/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_4/Tanh:y:0*
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
while/IdentityIdentitywhile/add_1:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_4/Tanh:y:0/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 
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
?
?
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_22235

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
?#
?
while_body_19098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_4_19120_0#
while_simple_rnn_cell_4_19122_0#
while_simple_rnn_cell_4_19124_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_4_19120!
while_simple_rnn_cell_4_19122!
while_simple_rnn_cell_4_19124??/while/simple_rnn_cell_4/StatefulPartitionedCall?
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
/while/simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_4_19120_0while_simple_rnn_cell_4_19122_0while_simple_rnn_cell_4_19124_0*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_1882421
/while/simple_rnn_cell_4/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_4/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_4/StatefulPartitionedCall:output:10^while/simple_rnn_cell_4/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_4_19120while_simple_rnn_cell_4_19120_0"@
while_simple_rnn_cell_4_19122while_simple_rnn_cell_4_19122_0"@
while_simple_rnn_cell_4_19124while_simple_rnn_cell_4_19124_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*?
_input_shapes.
,: : : : :??????????: : :::2b
/while/simple_rnn_cell_4/StatefulPartitionedCall/while/simple_rnn_cell_4/StatefulPartitionedCall: 
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
H
,__inference_activation_3_layer_call_fn_22112

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
GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_203542
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
?
?
,__inference_sequential_1_layer_call_fn_21063

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
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_204742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
1__inference_simple_rnn_cell_4_layer_call_fn_22204

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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_188242
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
?
?
,__inference_simple_rnn_2_layer_call_fn_21330

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
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_200262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_simple_rnn_3_layer_call_fn_22102
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_197902
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
while_cond_19214
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_19214___redundant_placeholder03
/while_while_cond_19214___redundant_placeholder13
/while_while_cond_19214___redundant_placeholder23
/while_while_cond_19214___redundant_placeholder3
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
?
E
)__inference_dropout_2_layer_call_fn_21610

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
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_200832
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
simple_rnn_2_while_cond_206176
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1M
Isimple_rnn_2_while_simple_rnn_2_while_cond_20617___redundant_placeholder0M
Isimple_rnn_2_while_simple_rnn_2_while_cond_20617___redundant_placeholder1M
Isimple_rnn_2_while_simple_rnn_2_while_cond_20617___redundant_placeholder2M
Isimple_rnn_2_while_simple_rnn_2_while_cond_20617___redundant_placeholder3
simple_rnn_2_while_identity
?
simple_rnn_2/while/LessLesssimple_rnn_2_while_placeholder4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_2/while/Less?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_2/while/Identity"C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0*A
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
?
?
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_19353

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
while_cond_21375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21375___redundant_placeholder03
/while_while_cond_21375___redundant_placeholder13
/while_while_cond_21375___redundant_placeholder23
/while_while_cond_21375___redundant_placeholder3
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
?
?
,__inference_sequential_1_layer_call_fn_20541
simple_rnn_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_205222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
,
_output_shapes
:??????????
,
_user_specified_namesimple_rnn_2_input
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_20474

inputs
simple_rnn_2_20450
simple_rnn_2_20452
simple_rnn_2_20454
simple_rnn_3_20459
simple_rnn_3_20461
simple_rnn_3_20463
dense_1_20468
dense_1_20470
identity??dense_1/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?$simple_rnn_3/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_20450simple_rnn_2_20452simple_rnn_2_20454*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_199142&
$simple_rnn_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_200612
activation_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_200782
dropout_2/PartitionedCall?
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0simple_rnn_3_20459simple_rnn_3_20461simple_rnn_3_20463*
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_202072&
$simple_rnn_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_203542
activation_3/PartitionedCall?
dropout_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_203712
dropout_3/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_1_20468dense_1_20470*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_204002!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_18841

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
`
D__inference_dropout_2_layer_call_and_return_conditional_losses_21595

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
:???????????2
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
:???????????2	
dropoutj
IdentityIdentitydropout:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_20522

inputs
simple_rnn_2_20498
simple_rnn_2_20500
simple_rnn_2_20502
simple_rnn_3_20507
simple_rnn_3_20509
simple_rnn_3_20511
dense_1_20516
dense_1_20518
identity??dense_1/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?$simple_rnn_3/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_20498simple_rnn_2_20500simple_rnn_2_20502*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_200262&
$simple_rnn_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_200612
activation_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_200832
dropout_2/PartitionedCall?
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0simple_rnn_3_20507simple_rnn_3_20509simple_rnn_3_20511*
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
GPU 2J 8? *P
fKRI
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_203192&
$simple_rnn_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_203542
activation_3/PartitionedCall?
dropout_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
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
GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_203762
dropout_3/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_1_20516dense_1_20518*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_204002!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?#
?
while_body_19610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_5_19632_0#
while_simple_rnn_cell_5_19634_0#
while_simple_rnn_cell_5_19636_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_5_19632!
while_simple_rnn_cell_5_19634!
while_simple_rnn_cell_5_19636??/while/simple_rnn_cell_5/StatefulPartitionedCall?
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
/while/simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_5_19632_0while_simple_rnn_cell_5_19634_0while_simple_rnn_cell_5_19636_0*
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
GPU 2J 8? *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_1933621
/while/simple_rnn_cell_5/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_5/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_5/StatefulPartitionedCall:output:10^while/simple_rnn_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_5_19632while_simple_rnn_cell_5_19632_0"@
while_simple_rnn_cell_5_19634while_simple_rnn_cell_5_19634_0"@
while_simple_rnn_cell_5_19636while_simple_rnn_cell_5_19636_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2b
/while/simple_rnn_cell_5/StatefulPartitionedCall/while/simple_rnn_cell_5/StatefulPartitionedCall: 
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

?
simple_rnn_2_while_cond_208566
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1M
Isimple_rnn_2_while_simple_rnn_2_while_cond_20856___redundant_placeholder0M
Isimple_rnn_2_while_simple_rnn_2_while_cond_20856___redundant_placeholder1M
Isimple_rnn_2_while_simple_rnn_2_while_cond_20856___redundant_placeholder2M
Isimple_rnn_2_while_simple_rnn_2_while_cond_20856___redundant_placeholder3
simple_rnn_2_while_identity
?
simple_rnn_2/while/LessLesssimple_rnn_2_while_placeholder4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: 2
simple_rnn_2/while/Less?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_2/while/Identity"C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0*A
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
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
V
simple_rnn_2_input@
$serving_default_simple_rnn_2_input:0??????????<
dense_11
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?6
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
		variables

trainable_variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?4
_tf_keras_sequential?3{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_2_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 240, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "simple_rnn_2_input"}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 240, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_rnn_layer?	{"class_name": "SimpleRNN", "name": "simple_rnn_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_2", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 5]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
cell

state_spec
	variables
trainable_variables
 regularization_losses
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_rnn_layer?	{"class_name": "SimpleRNN", "name": "simple_rnn_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 150]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 150]}}
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 240, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
0iter

1beta_1

2beta_2
	3decay
4learning_rate*m?+m?5m?6m?7m?8m?9m?:m?*v?+v?5v?6v?7v?8v?9v?:v?"
	optimizer
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
 "
trackable_list_wrapper
?
;metrics
		variables

trainable_variables
regularization_losses
<layer_regularization_losses
=non_trainable_variables
>layer_metrics

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
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_4", "trainable": true, "dtype": "float32", "units": 150, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
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
 "
trackable_list_wrapper
?
Dmetrics

Estates
	variables
trainable_variables
regularization_losses
Flayer_regularization_losses
Gnon_trainable_variables
Hlayer_metrics

Ilayers
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
Jmetrics
	variables
trainable_variables
regularization_losses

Klayers
Llayer_regularization_losses
Mlayer_metrics
Nnon_trainable_variables
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
Ometrics
	variables
trainable_variables
regularization_losses

Players
Qlayer_regularization_losses
Rlayer_metrics
Snon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

8kernel
9recurrent_kernel
:bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
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
 "
trackable_list_wrapper
?
Xmetrics

Ystates
	variables
trainable_variables
 regularization_losses
Zlayer_regularization_losses
[non_trainable_variables
\layer_metrics

]layers
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
^metrics
"	variables
#trainable_variables
$regularization_losses

_layers
`layer_regularization_losses
alayer_metrics
bnon_trainable_variables
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
cmetrics
&	variables
'trainable_variables
(regularization_losses

dlayers
elayer_regularization_losses
flayer_metrics
gnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	2?2dense_1/kernel
:?2dense_1/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
hmetrics
,	variables
-trainable_variables
.regularization_losses

ilayers
jlayer_regularization_losses
klayer_metrics
lnon_trainable_variables
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
8:6	?2%simple_rnn_2/simple_rnn_cell_4/kernel
C:A
??2/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel
2:0?2#simple_rnn_2/simple_rnn_cell_4/bias
8:6	?22%simple_rnn_3/simple_rnn_cell_5/kernel
A:?222/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel
1:/22#simple_rnn_3/simple_rnn_cell_5/bias
5
m0
n1
o2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
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
 "
trackable_list_wrapper
?
pmetrics
@	variables
Atrainable_variables
Bregularization_losses

qlayers
rlayer_regularization_losses
slayer_metrics
tnon_trainable_variables
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
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
 "
trackable_list_wrapper
?
umetrics
T	variables
Utrainable_variables
Vregularization_losses

vlayers
wlayer_regularization_losses
xlayer_metrics
ynon_trainable_variables
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
&:$	2?2Adam/dense_1/kernel/m
 :?2Adam/dense_1/bias/m
=:;	?2,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/m
H:F
??26Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/m
7:5?2*Adam/simple_rnn_2/simple_rnn_cell_4/bias/m
=:;	?22,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/m
F:D2226Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/m
6:422*Adam/simple_rnn_3/simple_rnn_cell_5/bias/m
&:$	2?2Adam/dense_1/kernel/v
 :?2Adam/dense_1/bias/v
=:;	?2,Adam/simple_rnn_2/simple_rnn_cell_4/kernel/v
H:F
??26Adam/simple_rnn_2/simple_rnn_cell_4/recurrent_kernel/v
7:5?2*Adam/simple_rnn_2/simple_rnn_cell_4/bias/v
=:;	?22,Adam/simple_rnn_3/simple_rnn_cell_5/kernel/v
F:D2226Adam/simple_rnn_3/simple_rnn_cell_5/recurrent_kernel/v
6:422*Adam/simple_rnn_3/simple_rnn_cell_5/bias/v
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_20811
G__inference_sequential_1_layer_call_and_return_conditional_losses_20444
G__inference_sequential_1_layer_call_and_return_conditional_losses_21042
G__inference_sequential_1_layer_call_and_return_conditional_losses_20417?
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
 __inference__wrapped_model_18775?
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
annotations? *6?3
1?.
simple_rnn_2_input??????????
?2?
,__inference_sequential_1_layer_call_fn_21063
,__inference_sequential_1_layer_call_fn_20493
,__inference_sequential_1_layer_call_fn_21084
,__inference_sequential_1_layer_call_fn_20541?
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
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21554
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21196
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21442
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21308?
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
?2?
,__inference_simple_rnn_2_layer_call_fn_21330
,__inference_simple_rnn_2_layer_call_fn_21319
,__inference_simple_rnn_2_layer_call_fn_21565
,__inference_simple_rnn_2_layer_call_fn_21576?
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
G__inference_activation_2_layer_call_and_return_conditional_losses_21581?
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
,__inference_activation_2_layer_call_fn_21586?
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
D__inference_dropout_2_layer_call_and_return_conditional_losses_21595
D__inference_dropout_2_layer_call_and_return_conditional_losses_21600?
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
)__inference_dropout_2_layer_call_fn_21605
)__inference_dropout_2_layer_call_fn_21610?
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
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_21834
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_21968
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_22080
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_21722?
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
?2?
,__inference_simple_rnn_3_layer_call_fn_21856
,__inference_simple_rnn_3_layer_call_fn_21845
,__inference_simple_rnn_3_layer_call_fn_22091
,__inference_simple_rnn_3_layer_call_fn_22102?
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
G__inference_activation_3_layer_call_and_return_conditional_losses_22107?
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
,__inference_activation_3_layer_call_fn_22112?
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
D__inference_dropout_3_layer_call_and_return_conditional_losses_22121
D__inference_dropout_3_layer_call_and_return_conditional_losses_22126?
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
)__inference_dropout_3_layer_call_fn_22136
)__inference_dropout_3_layer_call_fn_22131?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_22147?
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
'__inference_dense_1_layer_call_fn_22156?
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
#__inference_signature_wrapper_20572simple_rnn_2_input"?
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
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_22173
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_22190?
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
1__inference_simple_rnn_cell_4_layer_call_fn_22204
1__inference_simple_rnn_cell_4_layer_call_fn_22218?
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
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_22235
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_22252?
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
1__inference_simple_rnn_cell_5_layer_call_fn_22280
1__inference_simple_rnn_cell_5_layer_call_fn_22266?
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
 __inference__wrapped_model_18775?5768:9*+@?=
6?3
1?.
simple_rnn_2_input??????????
? "2?/
-
dense_1"?
dense_1???????????
G__inference_activation_2_layer_call_and_return_conditional_losses_21581d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
,__inference_activation_2_layer_call_fn_21586W5?2
+?(
&?#
inputs???????????
? "?????????????
G__inference_activation_3_layer_call_and_return_conditional_losses_22107X/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? {
,__inference_activation_3_layer_call_fn_22112K/?,
%?"
 ?
inputs?????????2
? "??????????2?
B__inference_dense_1_layer_call_and_return_conditional_losses_22147]*+/?,
%?"
 ?
inputs?????????2
? "&?#
?
0??????????
? {
'__inference_dense_1_layer_call_fn_22156P*+/?,
%?"
 ?
inputs?????????2
? "????????????
D__inference_dropout_2_layer_call_and_return_conditional_losses_21595h9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_21600h9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
)__inference_dropout_2_layer_call_fn_21605[9?6
/?,
&?#
inputs???????????
p
? "?????????????
)__inference_dropout_2_layer_call_fn_21610[9?6
/?,
&?#
inputs???????????
p 
? "?????????????
D__inference_dropout_3_layer_call_and_return_conditional_losses_22121\3?0
)?&
 ?
inputs?????????2
p
? "%?"
?
0?????????2
? ?
D__inference_dropout_3_layer_call_and_return_conditional_losses_22126\3?0
)?&
 ?
inputs?????????2
p 
? "%?"
?
0?????????2
? |
)__inference_dropout_3_layer_call_fn_22131O3?0
)?&
 ?
inputs?????????2
p
? "??????????2|
)__inference_dropout_3_layer_call_fn_22136O3?0
)?&
 ?
inputs?????????2
p 
? "??????????2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_20417|5768:9*+H?E
>?;
1?.
simple_rnn_2_input??????????
p

 
? "&?#
?
0??????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_20444|5768:9*+H?E
>?;
1?.
simple_rnn_2_input??????????
p 

 
? "&?#
?
0??????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_20811p5768:9*+<?9
2?/
%?"
inputs??????????
p

 
? "&?#
?
0??????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_21042p5768:9*+<?9
2?/
%?"
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
,__inference_sequential_1_layer_call_fn_20493o5768:9*+H?E
>?;
1?.
simple_rnn_2_input??????????
p

 
? "????????????
,__inference_sequential_1_layer_call_fn_20541o5768:9*+H?E
>?;
1?.
simple_rnn_2_input??????????
p 

 
? "????????????
,__inference_sequential_1_layer_call_fn_21063c5768:9*+<?9
2?/
%?"
inputs??????????
p

 
? "????????????
,__inference_sequential_1_layer_call_fn_21084c5768:9*+<?9
2?/
%?"
inputs??????????
p 

 
? "????????????
#__inference_signature_wrapper_20572?5768:9*+V?S
? 
L?I
G
simple_rnn_2_input1?.
simple_rnn_2_input??????????"2?/
-
dense_1"?
dense_1???????????
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21196t576@?=
6?3
%?"
inputs??????????

 
p

 
? "+?(
!?
0???????????
? ?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21308t576@?=
6?3
%?"
inputs??????????

 
p 

 
? "+?(
!?
0???????????
? ?
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21442?576O?L
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
G__inference_simple_rnn_2_layer_call_and_return_conditional_losses_21554?576O?L
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
,__inference_simple_rnn_2_layer_call_fn_21319g576@?=
6?3
%?"
inputs??????????

 
p

 
? "?????????????
,__inference_simple_rnn_2_layer_call_fn_21330g576@?=
6?3
%?"
inputs??????????

 
p 

 
? "?????????????
,__inference_simple_rnn_2_layer_call_fn_21565~576O?L
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
,__inference_simple_rnn_2_layer_call_fn_21576~576O?L
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
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_21722o8:9A?>
7?4
&?#
inputs???????????

 
p

 
? "%?"
?
0?????????2
? ?
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_21834o8:9A?>
7?4
&?#
inputs???????????

 
p 

 
? "%?"
?
0?????????2
? ?
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_21968~8:9P?M
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
G__inference_simple_rnn_3_layer_call_and_return_conditional_losses_22080~8:9P?M
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
,__inference_simple_rnn_3_layer_call_fn_21845b8:9A?>
7?4
&?#
inputs???????????

 
p

 
? "??????????2?
,__inference_simple_rnn_3_layer_call_fn_21856b8:9A?>
7?4
&?#
inputs???????????

 
p 

 
? "??????????2?
,__inference_simple_rnn_3_layer_call_fn_22091q8:9P?M
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
,__inference_simple_rnn_3_layer_call_fn_22102q8:9P?M
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
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_22173?576]?Z
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
L__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_22190?576]?Z
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
1__inference_simple_rnn_cell_4_layer_call_fn_22204?576]?Z
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
1__inference_simple_rnn_cell_4_layer_call_fn_22218?576]?Z
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
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_22235?8:9]?Z
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
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_22252?8:9]?Z
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
1__inference_simple_rnn_cell_5_layer_call_fn_22266?8:9]?Z
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
1__inference_simple_rnn_cell_5_layer_call_fn_22280?8:9]?Z
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