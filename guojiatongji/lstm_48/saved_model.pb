??
??
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
MLCLSTM
x"T
hidden_size
output_size
kernel"T
recurrent_kernel"T	
bias"T
output"T"
Ttype:
2"
return_sequencesbool( "
dropoutfloat"
bidirectionalbool( "

activationstringtanh
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.4.0-rc02&tf_macos-v0.1-alpha2-AS-67-gf3595294ab8??
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:20*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:20*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:0*
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
lstm_4/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namelstm_4/lstm_cell_8/kernel
?
-lstm_4/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_8/kernel*
_output_shapes
:	?*
dtype0
?
#lstm_4/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#lstm_4/lstm_cell_8/recurrent_kernel
?
7lstm_4/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_4/lstm_cell_8/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_4/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_4/lstm_cell_8/bias
?
+lstm_4/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_8/bias*
_output_shapes	
:?*
dtype0
?
lstm_5/lstm_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namelstm_5/lstm_cell_9/kernel
?
-lstm_5/lstm_cell_9/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_9/kernel*
_output_shapes
:	?*
dtype0
?
#lstm_5/lstm_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*4
shared_name%#lstm_5/lstm_cell_9/recurrent_kernel
?
7lstm_5/lstm_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_5/lstm_cell_9/recurrent_kernel*
_output_shapes
:	2?*
dtype0
?
lstm_5/lstm_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_5/lstm_cell_9/bias
?
+lstm_5/lstm_cell_9/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_9/bias*
_output_shapes	
:?*
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
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/m
?
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:20*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:20*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:0*
dtype0
?
 Adam/lstm_4/lstm_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_4/lstm_cell_8/kernel/m
?
4Adam/lstm_4/lstm_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_8/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_4/lstm_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_4/lstm_cell_8/recurrent_kernel/m
?
>Adam/lstm_4/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_8/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_4/lstm_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_4/lstm_cell_8/bias/m
?
2Adam/lstm_4/lstm_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_8/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_5/lstm_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_5/lstm_cell_9/kernel/m
?
4Adam/lstm_5/lstm_cell_9/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_9/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_5/lstm_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/lstm_5/lstm_cell_9/recurrent_kernel/m
?
>Adam/lstm_5/lstm_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_9/recurrent_kernel/m*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_5/lstm_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_5/lstm_cell_9/bias/m
?
2Adam/lstm_5/lstm_cell_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_9/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/v
?
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:20*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:20*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:0*
dtype0
?
 Adam/lstm_4/lstm_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_4/lstm_cell_8/kernel/v
?
4Adam/lstm_4/lstm_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_8/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_4/lstm_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_4/lstm_cell_8/recurrent_kernel/v
?
>Adam/lstm_4/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_8/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_4/lstm_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_4/lstm_cell_8/bias/v
?
2Adam/lstm_4/lstm_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_8/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_5/lstm_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_5/lstm_cell_9/kernel/v
?
4Adam/lstm_5/lstm_cell_9/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_9/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_5/lstm_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/lstm_5/lstm_cell_9/recurrent_kernel/v
?
>Adam/lstm_5/lstm_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_9/recurrent_kernel/v*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_5/lstm_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_5/lstm_cell_9/bias/v
?
2Adam/lstm_5/lstm_cell_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_9/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?;
value?;B?; B?;
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemlmmmnmo(mp)mq*mr+ms,mt-muvvvwvxvy(vz)v{*v|+v},v~-v
F
(0
)1
*2
3
4
+5
,6
-7
8
9
 
F
(0
)1
*2
3
4
+5
,6
-7
8
9
?

.layers
	variables
/metrics
regularization_losses
0layer_metrics
trainable_variables
1non_trainable_variables
2layer_regularization_losses
 
~

(kernel
)recurrent_kernel
*bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
 

(0
)1
*2
 

(0
)1
*2
?

7layers

8states
	variables
9metrics
regularization_losses
:layer_metrics
trainable_variables
;non_trainable_variables
<layer_regularization_losses
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

=layers
	variables
>metrics
regularization_losses
?layer_metrics
trainable_variables
@non_trainable_variables
Alayer_regularization_losses
~

+kernel
,recurrent_kernel
-bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
 

+0
,1
-2
 

+0
,1
-2
?

Flayers

Gstates
	variables
Hmetrics
regularization_losses
Ilayer_metrics
trainable_variables
Jnon_trainable_variables
Klayer_regularization_losses
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Llayers
	variables
Mmetrics
 regularization_losses
Nlayer_metrics
!trainable_variables
Onon_trainable_variables
Player_regularization_losses
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
US
VARIABLE_VALUElstm_4/lstm_cell_8/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_4/lstm_cell_8/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_4/lstm_cell_8/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_5/lstm_cell_9/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_5/lstm_cell_9/recurrent_kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_5/lstm_cell_9/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

Q0
R1
S2
 
 
 

(0
)1
*2
 

(0
)1
*2
?

Tlayers
3	variables
Umetrics
4regularization_losses
Vlayer_metrics
5trainable_variables
Wnon_trainable_variables
Xlayer_regularization_losses

0
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
+0
,1
-2
 

+0
,1
-2
?

Ylayers
B	variables
Zmetrics
Cregularization_losses
[layer_metrics
Dtrainable_variables
\non_trainable_variables
]layer_regularization_losses

0
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
	^total
	_count
`	variables
a	keras_api
D
	btotal
	ccount
d
_fn_kwargs
e	variables
f	keras_api
D
	gtotal
	hcount
i
_fn_kwargs
j	variables
k	keras_api
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
^0
_1

`	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

b0
c1

e	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

j	variables
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_4/lstm_cell_8/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_8/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_8/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_5/lstm_cell_9/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_5/lstm_cell_9/recurrent_kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_5/lstm_cell_9/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_4/lstm_cell_8/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_8/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_8/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_5/lstm_cell_9/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_5/lstm_cell_9/recurrent_kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_5/lstm_cell_9/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_4_inputPlaceholder*+
_output_shapes
:?????????`*
dtype0* 
shape:?????????`
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_4_inputlstm_4/lstm_cell_8/kernel#lstm_4/lstm_cell_8/recurrent_kernellstm_4/lstm_cell_8/biasdense_8/kerneldense_8/biaslstm_5/lstm_cell_9/kernel#lstm_5/lstm_cell_9/recurrent_kernellstm_5/lstm_cell_9/biasdense_9/kerneldense_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_71276
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_4/lstm_cell_8/kernel/Read/ReadVariableOp7lstm_4/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp+lstm_4/lstm_cell_8/bias/Read/ReadVariableOp-lstm_5/lstm_cell_9/kernel/Read/ReadVariableOp7lstm_5/lstm_cell_9/recurrent_kernel/Read/ReadVariableOp+lstm_5/lstm_cell_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_8/kernel/m/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_8/bias/m/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_9/kernel/m/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_9/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_9/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_8/kernel/v/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_8/bias/v/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_9/kernel/v/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_9/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_9/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
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
__inference__traced_save_72040
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_4/lstm_cell_8/kernel#lstm_4/lstm_cell_8/recurrent_kernellstm_4/lstm_cell_8/biaslstm_5/lstm_cell_9/kernel#lstm_5/lstm_cell_9/recurrent_kernellstm_5/lstm_cell_9/biastotalcounttotal_1count_1total_2count_2Adam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/m Adam/lstm_4/lstm_cell_8/kernel/m*Adam/lstm_4/lstm_cell_8/recurrent_kernel/mAdam/lstm_4/lstm_cell_8/bias/m Adam/lstm_5/lstm_cell_9/kernel/m*Adam/lstm_5/lstm_cell_9/recurrent_kernel/mAdam/lstm_5/lstm_cell_9/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v Adam/lstm_4/lstm_cell_8/kernel/v*Adam/lstm_4/lstm_cell_8/recurrent_kernel/vAdam/lstm_4/lstm_cell_8/bias/v Adam/lstm_5/lstm_cell_9/kernel/v*Adam/lstm_5/lstm_cell_9/recurrent_kernel/vAdam/lstm_5/lstm_cell_9/bias/v*5
Tin.
,2**
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
!__inference__traced_restore_72173??

?"
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_70793

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_7_layer_call_fn_71188
lstm_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_711652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????`
&
_user_specified_namelstm_4_input
?V
?
__inference__traced_save_72040
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_4_lstm_cell_8_kernel_read_readvariableopB
>savev2_lstm_4_lstm_cell_8_recurrent_kernel_read_readvariableop6
2savev2_lstm_4_lstm_cell_8_bias_read_readvariableop8
4savev2_lstm_5_lstm_cell_9_kernel_read_readvariableopB
>savev2_lstm_5_lstm_cell_9_recurrent_kernel_read_readvariableop6
2savev2_lstm_5_lstm_cell_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_8_kernel_m_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_8_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_8_bias_m_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_9_kernel_m_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_9_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_9_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_8_kernel_v_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_8_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_8_bias_v_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_9_kernel_v_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_9_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_9_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_4_lstm_cell_8_kernel_read_readvariableop>savev2_lstm_4_lstm_cell_8_recurrent_kernel_read_readvariableop2savev2_lstm_4_lstm_cell_8_bias_read_readvariableop4savev2_lstm_5_lstm_cell_9_kernel_read_readvariableop>savev2_lstm_5_lstm_cell_9_recurrent_kernel_read_readvariableop2savev2_lstm_5_lstm_cell_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop;savev2_adam_lstm_4_lstm_cell_8_kernel_m_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_8_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_4_lstm_cell_8_bias_m_read_readvariableop;savev2_adam_lstm_5_lstm_cell_9_kernel_m_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_9_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_5_lstm_cell_9_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop;savev2_adam_lstm_4_lstm_cell_8_kernel_v_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_8_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_4_lstm_cell_8_bias_v_read_readvariableop;savev2_adam_lstm_5_lstm_cell_9_kernel_v_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_9_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_5_lstm_cell_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
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
_input_shapes?
?: :	?::20:0: : : : : :	?:
??:?:	?:	2?:?: : : : : : :	?::20:0:	?:
??:?:	?:	2?:?:	?::20:0:	?:
??:?:	?:	2?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:20: 

_output_shapes
:0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:
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
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:20: 

_output_shapes
:0:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::$" 

_output_shapes

:20: #

_output_shapes
:0:%$!

_output_shapes
:	?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	?:%(!

_output_shapes
:	2?:!)

_output_shapes	
:?:*

_output_shapes
: 
?"
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71816
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?!
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_70672

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_5_layer_call_fn_71874
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
GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_708402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
&__inference_lstm_4_layer_call_fn_71666

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
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_709202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
&__inference_lstm_4_layer_call_fn_71565
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
GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_706272
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
? 
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_70920

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

?
B__inference_dense_9_layer_call_and_return_conditional_losses_71089

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:20*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????02
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????02

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
?x
?
 __inference__wrapped_model_70521
lstm_4_input7
3sequential_7_lstm_4_mlclstm_readvariableop_resource9
5sequential_7_lstm_4_mlclstm_readvariableop_1_resource9
5sequential_7_lstm_4_mlclstm_readvariableop_2_resource:
6sequential_7_dense_8_mlcmatmul_readvariableop_resource8
4sequential_7_dense_8_biasadd_readvariableop_resource7
3sequential_7_lstm_5_mlclstm_readvariableop_resource9
5sequential_7_lstm_5_mlclstm_readvariableop_1_resource9
5sequential_7_lstm_5_mlclstm_readvariableop_2_resource:
6sequential_7_dense_9_mlcmatmul_readvariableop_resource8
4sequential_7_dense_9_biasadd_readvariableop_resource
identity??+sequential_7/dense_8/BiasAdd/ReadVariableOp?-sequential_7/dense_8/MLCMatMul/ReadVariableOp?+sequential_7/dense_9/BiasAdd/ReadVariableOp?-sequential_7/dense_9/MLCMatMul/ReadVariableOp?*sequential_7/lstm_4/MLCLSTM/ReadVariableOp?,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_1?,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_2?*sequential_7/lstm_5/MLCLSTM/ReadVariableOp?,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_1?,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_2r
sequential_7/lstm_4/ShapeShapelstm_4_input*
T0*
_output_shapes
:2
sequential_7/lstm_4/Shape?
'sequential_7/lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/lstm_4/strided_slice/stack?
)sequential_7/lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_7/lstm_4/strided_slice/stack_1?
)sequential_7/lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_7/lstm_4/strided_slice/stack_2?
!sequential_7/lstm_4/strided_sliceStridedSlice"sequential_7/lstm_4/Shape:output:00sequential_7/lstm_4/strided_slice/stack:output:02sequential_7/lstm_4/strided_slice/stack_1:output:02sequential_7/lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_7/lstm_4/strided_slice?
sequential_7/lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
sequential_7/lstm_4/zeros/mul/y?
sequential_7/lstm_4/zeros/mulMul*sequential_7/lstm_4/strided_slice:output:0(sequential_7/lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_7/lstm_4/zeros/mul?
 sequential_7/lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_7/lstm_4/zeros/Less/y?
sequential_7/lstm_4/zeros/LessLess!sequential_7/lstm_4/zeros/mul:z:0)sequential_7/lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_4/zeros/Less?
"sequential_7/lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_7/lstm_4/zeros/packed/1?
 sequential_7/lstm_4/zeros/packedPack*sequential_7/lstm_4/strided_slice:output:0+sequential_7/lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_7/lstm_4/zeros/packed?
sequential_7/lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_7/lstm_4/zeros/Const?
sequential_7/lstm_4/zerosFill)sequential_7/lstm_4/zeros/packed:output:0(sequential_7/lstm_4/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_7/lstm_4/zeros?
!sequential_7/lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_7/lstm_4/zeros_1/mul/y?
sequential_7/lstm_4/zeros_1/mulMul*sequential_7/lstm_4/strided_slice:output:0*sequential_7/lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_7/lstm_4/zeros_1/mul?
"sequential_7/lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_7/lstm_4/zeros_1/Less/y?
 sequential_7/lstm_4/zeros_1/LessLess#sequential_7/lstm_4/zeros_1/mul:z:0+sequential_7/lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_4/zeros_1/Less?
$sequential_7/lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential_7/lstm_4/zeros_1/packed/1?
"sequential_7/lstm_4/zeros_1/packedPack*sequential_7/lstm_4/strided_slice:output:0-sequential_7/lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_7/lstm_4/zeros_1/packed?
!sequential_7/lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_7/lstm_4/zeros_1/Const?
sequential_7/lstm_4/zeros_1Fill+sequential_7/lstm_4/zeros_1/packed:output:0*sequential_7/lstm_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_7/lstm_4/zeros_1?
'sequential_7/lstm_4/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2)
'sequential_7/lstm_4/MLCLSTM/hidden_size?
'sequential_7/lstm_4/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2)
'sequential_7/lstm_4/MLCLSTM/output_size?
*sequential_7/lstm_4/MLCLSTM/ReadVariableOpReadVariableOp3sequential_7_lstm_4_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_7/lstm_4/MLCLSTM/ReadVariableOp?
,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_1ReadVariableOp5sequential_7_lstm_4_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02.
,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_1?
,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_2ReadVariableOp5sequential_7_lstm_4_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02.
,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_2?
sequential_7/lstm_4/MLCLSTMMLCLSTMlstm_4_input0sequential_7/lstm_4/MLCLSTM/hidden_size:output:00sequential_7/lstm_4/MLCLSTM/output_size:output:02sequential_7/lstm_4/MLCLSTM/ReadVariableOp:value:04sequential_7/lstm_4/MLCLSTM/ReadVariableOp_1:value:04sequential_7/lstm_4/MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2
sequential_7/lstm_4/MLCLSTM?
-sequential_7/dense_8/MLCMatMul/ReadVariableOpReadVariableOp6sequential_7_dense_8_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-sequential_7/dense_8/MLCMatMul/ReadVariableOp?
sequential_7/dense_8/MLCMatMul	MLCMatMul$sequential_7/lstm_4/MLCLSTM:output:05sequential_7/dense_8/MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2 
sequential_7/dense_8/MLCMatMul?
+sequential_7/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_7/dense_8/BiasAdd/ReadVariableOp?
sequential_7/dense_8/BiasAddBiasAdd(sequential_7/dense_8/MLCMatMul:product:03sequential_7/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
sequential_7/dense_8/BiasAdd?
sequential_7/dense_8/TanhTanh%sequential_7/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
sequential_7/dense_8/Tanh?
sequential_7/lstm_5/ShapeShapesequential_7/dense_8/Tanh:y:0*
T0*
_output_shapes
:2
sequential_7/lstm_5/Shape?
'sequential_7/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/lstm_5/strided_slice/stack?
)sequential_7/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_7/lstm_5/strided_slice/stack_1?
)sequential_7/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_7/lstm_5/strided_slice/stack_2?
!sequential_7/lstm_5/strided_sliceStridedSlice"sequential_7/lstm_5/Shape:output:00sequential_7/lstm_5/strided_slice/stack:output:02sequential_7/lstm_5/strided_slice/stack_1:output:02sequential_7/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_7/lstm_5/strided_slice?
sequential_7/lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
sequential_7/lstm_5/zeros/mul/y?
sequential_7/lstm_5/zeros/mulMul*sequential_7/lstm_5/strided_slice:output:0(sequential_7/lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_7/lstm_5/zeros/mul?
 sequential_7/lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_7/lstm_5/zeros/Less/y?
sequential_7/lstm_5/zeros/LessLess!sequential_7/lstm_5/zeros/mul:z:0)sequential_7/lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_5/zeros/Less?
"sequential_7/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"sequential_7/lstm_5/zeros/packed/1?
 sequential_7/lstm_5/zeros/packedPack*sequential_7/lstm_5/strided_slice:output:0+sequential_7/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_7/lstm_5/zeros/packed?
sequential_7/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_7/lstm_5/zeros/Const?
sequential_7/lstm_5/zerosFill)sequential_7/lstm_5/zeros/packed:output:0(sequential_7/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_7/lstm_5/zeros?
!sequential_7/lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22#
!sequential_7/lstm_5/zeros_1/mul/y?
sequential_7/lstm_5/zeros_1/mulMul*sequential_7/lstm_5/strided_slice:output:0*sequential_7/lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_7/lstm_5/zeros_1/mul?
"sequential_7/lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_7/lstm_5/zeros_1/Less/y?
 sequential_7/lstm_5/zeros_1/LessLess#sequential_7/lstm_5/zeros_1/mul:z:0+sequential_7/lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_5/zeros_1/Less?
$sequential_7/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22&
$sequential_7/lstm_5/zeros_1/packed/1?
"sequential_7/lstm_5/zeros_1/packedPack*sequential_7/lstm_5/strided_slice:output:0-sequential_7/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_7/lstm_5/zeros_1/packed?
!sequential_7/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_7/lstm_5/zeros_1/Const?
sequential_7/lstm_5/zeros_1Fill+sequential_7/lstm_5/zeros_1/packed:output:0*sequential_7/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_7/lstm_5/zeros_1?
'sequential_7/lstm_5/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22)
'sequential_7/lstm_5/MLCLSTM/hidden_size?
'sequential_7/lstm_5/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22)
'sequential_7/lstm_5/MLCLSTM/output_size?
*sequential_7/lstm_5/MLCLSTM/ReadVariableOpReadVariableOp3sequential_7_lstm_5_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_7/lstm_5/MLCLSTM/ReadVariableOp?
,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_1ReadVariableOp5sequential_7_lstm_5_mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02.
,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_1?
,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_2ReadVariableOp5sequential_7_lstm_5_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02.
,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_2?
sequential_7/lstm_5/MLCLSTMMLCLSTMsequential_7/dense_8/Tanh:y:00sequential_7/lstm_5/MLCLSTM/hidden_size:output:00sequential_7/lstm_5/MLCLSTM/output_size:output:02sequential_7/lstm_5/MLCLSTM/ReadVariableOp:value:04sequential_7/lstm_5/MLCLSTM/ReadVariableOp_1:value:04sequential_7/lstm_5/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2
sequential_7/lstm_5/MLCLSTM?
!sequential_7/lstm_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2#
!sequential_7/lstm_5/Reshape/shape?
sequential_7/lstm_5/ReshapeReshape$sequential_7/lstm_5/MLCLSTM:output:0*sequential_7/lstm_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22
sequential_7/lstm_5/Reshape?
-sequential_7/dense_9/MLCMatMul/ReadVariableOpReadVariableOp6sequential_7_dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:20*
dtype02/
-sequential_7/dense_9/MLCMatMul/ReadVariableOp?
sequential_7/dense_9/MLCMatMul	MLCMatMul$sequential_7/lstm_5/Reshape:output:05sequential_7/dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02 
sequential_7/dense_9/MLCMatMul?
+sequential_7/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_9_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02-
+sequential_7/dense_9/BiasAdd/ReadVariableOp?
sequential_7/dense_9/BiasAddBiasAdd(sequential_7/dense_9/MLCMatMul:product:03sequential_7/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
sequential_7/dense_9/BiasAdd?
sequential_7/dense_9/TanhTanh%sequential_7/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
sequential_7/dense_9/Tanh?
IdentityIdentitysequential_7/dense_9/Tanh:y:0,^sequential_7/dense_8/BiasAdd/ReadVariableOp.^sequential_7/dense_8/MLCMatMul/ReadVariableOp,^sequential_7/dense_9/BiasAdd/ReadVariableOp.^sequential_7/dense_9/MLCMatMul/ReadVariableOp+^sequential_7/lstm_4/MLCLSTM/ReadVariableOp-^sequential_7/lstm_4/MLCLSTM/ReadVariableOp_1-^sequential_7/lstm_4/MLCLSTM/ReadVariableOp_2+^sequential_7/lstm_5/MLCLSTM/ReadVariableOp-^sequential_7/lstm_5/MLCLSTM/ReadVariableOp_1-^sequential_7/lstm_5/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2Z
+sequential_7/dense_8/BiasAdd/ReadVariableOp+sequential_7/dense_8/BiasAdd/ReadVariableOp2^
-sequential_7/dense_8/MLCMatMul/ReadVariableOp-sequential_7/dense_8/MLCMatMul/ReadVariableOp2Z
+sequential_7/dense_9/BiasAdd/ReadVariableOp+sequential_7/dense_9/BiasAdd/ReadVariableOp2^
-sequential_7/dense_9/MLCMatMul/ReadVariableOp-sequential_7/dense_9/MLCMatMul/ReadVariableOp2X
*sequential_7/lstm_4/MLCLSTM/ReadVariableOp*sequential_7/lstm_4/MLCLSTM/ReadVariableOp2\
,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_1,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_12\
,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_2,sequential_7/lstm_4/MLCLSTM/ReadVariableOp_22X
*sequential_7/lstm_5/MLCLSTM/ReadVariableOp*sequential_7/lstm_5/MLCLSTM/ReadVariableOp2\
,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_1,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_12\
,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_2,sequential_7/lstm_5/MLCLSTM/ReadVariableOp_2:Y U
+
_output_shapes
:?????????`
&
_user_specified_namelstm_4_input
?"
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71048

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71106
lstm_4_input
lstm_4_70943
lstm_4_70945
lstm_4_70947
dense_8_70972
dense_8_70974
lstm_5_71071
lstm_5_71073
lstm_5_71075
dense_9_71100
dense_9_71102
identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
lstm_4/StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputlstm_4_70943lstm_4_70945lstm_4_70947*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_708862 
lstm_4/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_8_70972dense_8_70974*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_709612!
dense_8/StatefulPartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0lstm_5_71071lstm_5_71073lstm_5_71075*
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
GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_710122 
lstm_5/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_9_71100dense_9_71102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_710892!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????`
&
_user_specified_namelstm_4_input
?
?
#__inference_signature_wrapper_71276
lstm_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_705212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????`
&
_user_specified_namelstm_4_input
?!
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_71554
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?!
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_70627

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?^
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71436

inputs*
&lstm_4_mlclstm_readvariableop_resource,
(lstm_4_mlclstm_readvariableop_1_resource,
(lstm_4_mlclstm_readvariableop_2_resource-
)dense_8_mlcmatmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&lstm_5_mlclstm_readvariableop_resource,
(lstm_5_mlclstm_readvariableop_1_resource,
(lstm_5_mlclstm_readvariableop_2_resource-
)dense_9_mlcmatmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity??dense_8/BiasAdd/ReadVariableOp? dense_8/MLCMatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/MLCMatMul/ReadVariableOp?lstm_4/MLCLSTM/ReadVariableOp?lstm_4/MLCLSTM/ReadVariableOp_1?lstm_4/MLCLSTM/ReadVariableOp_2?lstm_5/MLCLSTM/ReadVariableOp?lstm_5/MLCLSTM/ReadVariableOp_1?lstm_5/MLCLSTM/ReadVariableOp_2R
lstm_4/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_4/Shape?
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice/stack?
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_1?
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_2?
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slicek
lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros/mul/y?
lstm_4/zeros/mulMullstm_4/strided_slice:output:0lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/mulm
lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros/Less/y?
lstm_4/zeros/LessLesslstm_4/zeros/mul:z:0lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/Lessq
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros/packed/1?
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros/packedm
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros/Const?
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_4/zeroso
lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros_1/mul/y?
lstm_4/zeros_1/mulMullstm_4/strided_slice:output:0lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/mulq
lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros_1/Less/y?
lstm_4/zeros_1/LessLesslstm_4/zeros_1/mul:z:0lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/Lessu
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros_1/packed/1?
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros_1/packedq
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros_1/Const?
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_4/zeros_1{
lstm_4/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/MLCLSTM/hidden_size{
lstm_4/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/MLCLSTM/output_size?
lstm_4/MLCLSTM/ReadVariableOpReadVariableOp&lstm_4_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
lstm_4/MLCLSTM/ReadVariableOp?
lstm_4/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_4_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02!
lstm_4/MLCLSTM/ReadVariableOp_1?
lstm_4/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_4_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_4/MLCLSTM/ReadVariableOp_2?
lstm_4/MLCLSTMMLCLSTMinputs#lstm_4/MLCLSTM/hidden_size:output:0#lstm_4/MLCLSTM/output_size:output:0%lstm_4/MLCLSTM/ReadVariableOp:value:0'lstm_4/MLCLSTM/ReadVariableOp_1:value:0'lstm_4/MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2
lstm_4/MLCLSTM?
 dense_8/MLCMatMul/ReadVariableOpReadVariableOp)dense_8_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_8/MLCMatMul/ReadVariableOp?
dense_8/MLCMatMul	MLCMatMullstm_4/MLCLSTM:output:0(dense_8/MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2
dense_8/MLCMatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MLCMatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
dense_8/BiasAddt
dense_8/TanhTanhdense_8/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
dense_8/Tanh\
lstm_5/ShapeShapedense_8/Tanh:y:0*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zeros_1z
lstm_5/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/MLCLSTM/hidden_sizez
lstm_5/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/MLCLSTM/output_size?
lstm_5/MLCLSTM/ReadVariableOpReadVariableOp&lstm_5_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
lstm_5/MLCLSTM/ReadVariableOp?
lstm_5/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_5_mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02!
lstm_5/MLCLSTM/ReadVariableOp_1?
lstm_5/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_5_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_5/MLCLSTM/ReadVariableOp_2?
lstm_5/MLCLSTMMLCLSTMdense_8/Tanh:y:0#lstm_5/MLCLSTM/hidden_size:output:0#lstm_5/MLCLSTM/output_size:output:0%lstm_5/MLCLSTM/ReadVariableOp:value:0'lstm_5/MLCLSTM/ReadVariableOp_1:value:0'lstm_5/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2
lstm_5/MLCLSTM}
lstm_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
lstm_5/Reshape/shape?
lstm_5/ReshapeReshapelstm_5/MLCLSTM:output:0lstm_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/Reshape?
 dense_9/MLCMatMul/ReadVariableOpReadVariableOp)dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:20*
dtype02"
 dense_9/MLCMatMul/ReadVariableOp?
dense_9/MLCMatMul	MLCMatMullstm_5/Reshape:output:0(dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_9/MLCMatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MLCMatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_9/BiasAddp
dense_9/TanhTanhdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
dense_9/Tanh?
IdentityIdentitydense_9/Tanh:y:0^dense_8/BiasAdd/ReadVariableOp!^dense_8/MLCMatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/MLCMatMul/ReadVariableOp^lstm_4/MLCLSTM/ReadVariableOp ^lstm_4/MLCLSTM/ReadVariableOp_1 ^lstm_4/MLCLSTM/ReadVariableOp_2^lstm_5/MLCLSTM/ReadVariableOp ^lstm_5/MLCLSTM/ReadVariableOp_1 ^lstm_5/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/MLCMatMul/ReadVariableOp dense_8/MLCMatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/MLCMatMul/ReadVariableOp dense_9/MLCMatMul/ReadVariableOp2>
lstm_4/MLCLSTM/ReadVariableOplstm_4/MLCLSTM/ReadVariableOp2B
lstm_4/MLCLSTM/ReadVariableOp_1lstm_4/MLCLSTM/ReadVariableOp_12B
lstm_4/MLCLSTM/ReadVariableOp_2lstm_4/MLCLSTM/ReadVariableOp_22>
lstm_5/MLCLSTM/ReadVariableOplstm_5/MLCLSTM/ReadVariableOp2B
lstm_5/MLCLSTM/ReadVariableOp_1lstm_5/MLCLSTM/ReadVariableOp_12B
lstm_5/MLCLSTM/ReadVariableOp_2lstm_5/MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
|
'__inference_dense_8_layer_call_fn_71686

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
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_709612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????`?::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????`?
 
_user_specified_nameinputs
?
?
&__inference_lstm_5_layer_call_fn_71769

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
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
GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_710122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?"
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71758

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71218

inputs
lstm_4_71193
lstm_4_71195
lstm_4_71197
dense_8_71200
dense_8_71202
lstm_5_71205
lstm_5_71207
lstm_5_71209
dense_9_71212
dense_9_71214
identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputslstm_4_71193lstm_4_71195lstm_4_71197*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_709202 
lstm_4/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_8_71200dense_8_71202*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_709612!
dense_8/StatefulPartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0lstm_5_71205lstm_5_71207lstm_5_71209*
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
GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_710482 
lstm_5/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_9_71212dense_9_71214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_710892!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
? 
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_70886

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71134
lstm_4_input
lstm_4_71109
lstm_4_71111
lstm_4_71113
dense_8_71116
dense_8_71118
lstm_5_71121
lstm_5_71123
lstm_5_71125
dense_9_71128
dense_9_71130
identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
lstm_4/StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputlstm_4_71109lstm_4_71111lstm_4_71113*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_709202 
lstm_4/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_8_71116dense_8_71118*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_709612!
dense_8/StatefulPartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0lstm_5_71121lstm_5_71123lstm_5_71125*
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
GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_710482 
lstm_5/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_9_71128dense_9_71130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_710892!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????`
&
_user_specified_namelstm_4_input
?"
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71722

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?"
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71852
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
|
'__inference_dense_9_layer_call_fn_71894

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
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_710892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
&__inference_lstm_4_layer_call_fn_71655

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
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_708862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?"
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71012

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
? 
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_71644

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
&__inference_lstm_5_layer_call_fn_71780

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
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
GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_710482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
,__inference_sequential_7_layer_call_fn_71241
lstm_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_712182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????`
&
_user_specified_namelstm_4_input
?

?
B__inference_dense_8_layer_call_and_return_conditional_losses_70961

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????`?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:T P
,
_output_shapes
:?????????`?
 
_user_specified_nameinputs
?
?
,__inference_sequential_7_layer_call_fn_71486

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_712182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?"
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_70840

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_4_layer_call_fn_71576
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
GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_706722
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
խ
?
!__inference__traced_restore_72173
file_prefix#
assignvariableop_dense_8_kernel#
assignvariableop_1_dense_8_bias%
!assignvariableop_2_dense_9_kernel#
assignvariableop_3_dense_9_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate0
,assignvariableop_9_lstm_4_lstm_cell_8_kernel;
7assignvariableop_10_lstm_4_lstm_cell_8_recurrent_kernel/
+assignvariableop_11_lstm_4_lstm_cell_8_bias1
-assignvariableop_12_lstm_5_lstm_cell_9_kernel;
7assignvariableop_13_lstm_5_lstm_cell_9_recurrent_kernel/
+assignvariableop_14_lstm_5_lstm_cell_9_bias
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1
assignvariableop_19_total_2
assignvariableop_20_count_2-
)assignvariableop_21_adam_dense_8_kernel_m+
'assignvariableop_22_adam_dense_8_bias_m-
)assignvariableop_23_adam_dense_9_kernel_m+
'assignvariableop_24_adam_dense_9_bias_m8
4assignvariableop_25_adam_lstm_4_lstm_cell_8_kernel_mB
>assignvariableop_26_adam_lstm_4_lstm_cell_8_recurrent_kernel_m6
2assignvariableop_27_adam_lstm_4_lstm_cell_8_bias_m8
4assignvariableop_28_adam_lstm_5_lstm_cell_9_kernel_mB
>assignvariableop_29_adam_lstm_5_lstm_cell_9_recurrent_kernel_m6
2assignvariableop_30_adam_lstm_5_lstm_cell_9_bias_m-
)assignvariableop_31_adam_dense_8_kernel_v+
'assignvariableop_32_adam_dense_8_bias_v-
)assignvariableop_33_adam_dense_9_kernel_v+
'assignvariableop_34_adam_dense_9_bias_v8
4assignvariableop_35_adam_lstm_4_lstm_cell_8_kernel_vB
>assignvariableop_36_adam_lstm_4_lstm_cell_8_recurrent_kernel_v6
2assignvariableop_37_adam_lstm_4_lstm_cell_8_bias_v8
4assignvariableop_38_adam_lstm_5_lstm_cell_9_kernel_vB
>assignvariableop_39_adam_lstm_5_lstm_cell_9_recurrent_kernel_v6
2assignvariableop_40_adam_lstm_5_lstm_cell_9_bias_v
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_4_lstm_cell_8_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_lstm_4_lstm_cell_8_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_4_lstm_cell_8_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_5_lstm_cell_9_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp7assignvariableop_13_lstm_5_lstm_cell_9_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp+assignvariableop_14_lstm_5_lstm_cell_9_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_8_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_8_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_9_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_9_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_lstm_4_lstm_cell_8_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_lstm_4_lstm_cell_8_recurrent_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_lstm_4_lstm_cell_8_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_5_lstm_cell_9_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_lstm_5_lstm_cell_9_recurrent_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_lstm_5_lstm_cell_9_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_8_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_8_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_9_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_9_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_lstm_4_lstm_cell_8_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp>assignvariableop_36_adam_lstm_4_lstm_cell_8_recurrent_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_lstm_4_lstm_cell_8_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp4assignvariableop_38_adam_lstm_5_lstm_cell_9_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp>assignvariableop_39_adam_lstm_5_lstm_cell_9_recurrent_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp2assignvariableop_40_adam_lstm_5_lstm_cell_9_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41?
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
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
?
?
,__inference_sequential_7_layer_call_fn_71461

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_711652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
? 
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_71610

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*,
_output_shapes
:?????????`?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71165

inputs
lstm_4_71140
lstm_4_71142
lstm_4_71144
dense_8_71147
dense_8_71149
lstm_5_71152
lstm_5_71154
lstm_5_71156
dense_9_71159
dense_9_71161
identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputslstm_4_71140lstm_4_71142lstm_4_71144*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????`?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_708862 
lstm_4/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_8_71147dense_8_71149*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_709612!
dense_8/StatefulPartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0lstm_5_71152lstm_5_71154lstm_5_71156*
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
GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_710122 
lstm_5/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_9_71159dense_9_71161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_710892!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?!
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_71520
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
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
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?^
?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71356

inputs*
&lstm_4_mlclstm_readvariableop_resource,
(lstm_4_mlclstm_readvariableop_1_resource,
(lstm_4_mlclstm_readvariableop_2_resource-
)dense_8_mlcmatmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&lstm_5_mlclstm_readvariableop_resource,
(lstm_5_mlclstm_readvariableop_1_resource,
(lstm_5_mlclstm_readvariableop_2_resource-
)dense_9_mlcmatmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity??dense_8/BiasAdd/ReadVariableOp? dense_8/MLCMatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/MLCMatMul/ReadVariableOp?lstm_4/MLCLSTM/ReadVariableOp?lstm_4/MLCLSTM/ReadVariableOp_1?lstm_4/MLCLSTM/ReadVariableOp_2?lstm_5/MLCLSTM/ReadVariableOp?lstm_5/MLCLSTM/ReadVariableOp_1?lstm_5/MLCLSTM/ReadVariableOp_2R
lstm_4/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_4/Shape?
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice/stack?
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_1?
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_2?
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slicek
lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros/mul/y?
lstm_4/zeros/mulMullstm_4/strided_slice:output:0lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/mulm
lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros/Less/y?
lstm_4/zeros/LessLesslstm_4/zeros/mul:z:0lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/Lessq
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros/packed/1?
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros/packedm
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros/Const?
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_4/zeroso
lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros_1/mul/y?
lstm_4/zeros_1/mulMullstm_4/strided_slice:output:0lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/mulq
lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros_1/Less/y?
lstm_4/zeros_1/LessLesslstm_4/zeros_1/mul:z:0lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/Lessu
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros_1/packed/1?
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros_1/packedq
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros_1/Const?
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_4/zeros_1{
lstm_4/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/MLCLSTM/hidden_size{
lstm_4/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/MLCLSTM/output_size?
lstm_4/MLCLSTM/ReadVariableOpReadVariableOp&lstm_4_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
lstm_4/MLCLSTM/ReadVariableOp?
lstm_4/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_4_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02!
lstm_4/MLCLSTM/ReadVariableOp_1?
lstm_4/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_4_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_4/MLCLSTM/ReadVariableOp_2?
lstm_4/MLCLSTMMLCLSTMinputs#lstm_4/MLCLSTM/hidden_size:output:0#lstm_4/MLCLSTM/output_size:output:0%lstm_4/MLCLSTM/ReadVariableOp:value:0'lstm_4/MLCLSTM/ReadVariableOp_1:value:0'lstm_4/MLCLSTM/ReadVariableOp_2:value:0*
T0*,
_output_shapes
:?????????`?*
dropout%    *
return_sequences(2
lstm_4/MLCLSTM?
 dense_8/MLCMatMul/ReadVariableOpReadVariableOp)dense_8_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_8/MLCMatMul/ReadVariableOp?
dense_8/MLCMatMul	MLCMatMullstm_4/MLCLSTM:output:0(dense_8/MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2
dense_8/MLCMatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MLCMatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2
dense_8/BiasAddt
dense_8/TanhTanhdense_8/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
dense_8/Tanh\
lstm_5/ShapeShapedense_8/Tanh:y:0*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zeros_1z
lstm_5/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/MLCLSTM/hidden_sizez
lstm_5/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/MLCLSTM/output_size?
lstm_5/MLCLSTM/ReadVariableOpReadVariableOp&lstm_5_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
lstm_5/MLCLSTM/ReadVariableOp?
lstm_5/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_5_mlclstm_readvariableop_1_resource*
_output_shapes
:	2?*
dtype02!
lstm_5/MLCLSTM/ReadVariableOp_1?
lstm_5/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_5_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_5/MLCLSTM/ReadVariableOp_2?
lstm_5/MLCLSTMMLCLSTMdense_8/Tanh:y:0#lstm_5/MLCLSTM/hidden_size:output:0#lstm_5/MLCLSTM/output_size:output:0%lstm_5/MLCLSTM/ReadVariableOp:value:0'lstm_5/MLCLSTM/ReadVariableOp_1:value:0'lstm_5/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????2*
dropout%    2
lstm_5/MLCLSTM}
lstm_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
lstm_5/Reshape/shape?
lstm_5/ReshapeReshapelstm_5/MLCLSTM:output:0lstm_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/Reshape?
 dense_9/MLCMatMul/ReadVariableOpReadVariableOp)dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:20*
dtype02"
 dense_9/MLCMatMul/ReadVariableOp?
dense_9/MLCMatMul	MLCMatMullstm_5/Reshape:output:0(dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_9/MLCMatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MLCMatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense_9/BiasAddp
dense_9/TanhTanhdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02
dense_9/Tanh?
IdentityIdentitydense_9/Tanh:y:0^dense_8/BiasAdd/ReadVariableOp!^dense_8/MLCMatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/MLCMatMul/ReadVariableOp^lstm_4/MLCLSTM/ReadVariableOp ^lstm_4/MLCLSTM/ReadVariableOp_1 ^lstm_4/MLCLSTM/ReadVariableOp_2^lstm_5/MLCLSTM/ReadVariableOp ^lstm_5/MLCLSTM/ReadVariableOp_1 ^lstm_5/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????`::::::::::2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/MLCMatMul/ReadVariableOp dense_8/MLCMatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/MLCMatMul/ReadVariableOp dense_9/MLCMatMul/ReadVariableOp2>
lstm_4/MLCLSTM/ReadVariableOplstm_4/MLCLSTM/ReadVariableOp2B
lstm_4/MLCLSTM/ReadVariableOp_1lstm_4/MLCLSTM/ReadVariableOp_12B
lstm_4/MLCLSTM/ReadVariableOp_2lstm_4/MLCLSTM/ReadVariableOp_22>
lstm_5/MLCLSTM/ReadVariableOplstm_5/MLCLSTM/ReadVariableOp2B
lstm_5/MLCLSTM/ReadVariableOp_1lstm_5/MLCLSTM/ReadVariableOp_12B
lstm_5/MLCLSTM/ReadVariableOp_2lstm_5/MLCLSTM/ReadVariableOp_2:S O
+
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

?
B__inference_dense_9_layer_call_and_return_conditional_losses_71885

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:20*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????02
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????02

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
?

?
B__inference_dense_8_layer_call_and_return_conditional_losses_71677

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`*

input_rank2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????`2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:?????????`2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????`?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:T P
,
_output_shapes
:?????????`?
 
_user_specified_nameinputs
?
?
&__inference_lstm_5_layer_call_fn_71863
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
GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_707932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
lstm_4_input9
serving_default_lstm_4_input:0?????????`;
dense_90
StatefulPartitionedCall:0?????????0tensorflow/serving/predict:??
?8
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?5
_tf_keras_sequential?5{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_4_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_4_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "lstm_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 5]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 150, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 5]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 150]}}
?
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 20]}}
?

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemlmmmnmo(mp)mq*mr+ms,mt-muvvvwvxvy(vz)v{*v|+v},v~-v"
	optimizer
f
(0
)1
*2
3
4
+5
,6
-7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
(0
)1
*2
3
4
+5
,6
-7
8
9"
trackable_list_wrapper
?

.layers
	variables
/metrics
regularization_losses
0layer_metrics
trainable_variables
1non_trainable_variables
2layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

(kernel
)recurrent_kernel
*bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_8", "trainable": true, "dtype": "float32", "units": 150, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
?

7layers

8states
	variables
9metrics
regularization_losses
:layer_metrics
trainable_variables
;non_trainable_variables
<layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_8/kernel
:2dense_8/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

=layers
	variables
>metrics
regularization_losses
?layer_metrics
trainable_variables
@non_trainable_variables
Alayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

+kernel
,recurrent_kernel
-bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_9", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?

Flayers

Gstates
	variables
Hmetrics
regularization_losses
Ilayer_metrics
trainable_variables
Jnon_trainable_variables
Klayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :202dense_9/kernel
:02dense_9/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Llayers
	variables
Mmetrics
 regularization_losses
Nlayer_metrics
!trainable_variables
Onon_trainable_variables
Player_regularization_losses
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
,:*	?2lstm_4/lstm_cell_8/kernel
7:5
??2#lstm_4/lstm_cell_8/recurrent_kernel
&:$?2lstm_4/lstm_cell_8/bias
,:*	?2lstm_5/lstm_cell_9/kernel
6:4	2?2#lstm_5/lstm_cell_9/recurrent_kernel
&:$?2lstm_5/lstm_cell_9/bias
<
0
1
2
3"
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
?

Tlayers
3	variables
Umetrics
4regularization_losses
Vlayer_metrics
5trainable_variables
Wnon_trainable_variables
Xlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?

Ylayers
B	variables
Zmetrics
Cregularization_losses
[layer_metrics
Dtrainable_variables
\non_trainable_variables
]layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
?
	^total
	_count
`	variables
a	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	btotal
	ccount
d
_fn_kwargs
e	variables
f	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
?
	gtotal
	hcount
i
_fn_kwargs
j	variables
k	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}
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
:  (2total
:  (2count
.
^0
_1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
g0
h1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
&:$	?2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
%:#202Adam/dense_9/kernel/m
:02Adam/dense_9/bias/m
1:/	?2 Adam/lstm_4/lstm_cell_8/kernel/m
<::
??2*Adam/lstm_4/lstm_cell_8/recurrent_kernel/m
+:)?2Adam/lstm_4/lstm_cell_8/bias/m
1:/	?2 Adam/lstm_5/lstm_cell_9/kernel/m
;:9	2?2*Adam/lstm_5/lstm_cell_9/recurrent_kernel/m
+:)?2Adam/lstm_5/lstm_cell_9/bias/m
&:$	?2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
%:#202Adam/dense_9/kernel/v
:02Adam/dense_9/bias/v
1:/	?2 Adam/lstm_4/lstm_cell_8/kernel/v
<::
??2*Adam/lstm_4/lstm_cell_8/recurrent_kernel/v
+:)?2Adam/lstm_4/lstm_cell_8/bias/v
1:/	?2 Adam/lstm_5/lstm_cell_9/kernel/v
;:9	2?2*Adam/lstm_5/lstm_cell_9/recurrent_kernel/v
+:)?2Adam/lstm_5/lstm_cell_9/bias/v
?2?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71356
G__inference_sequential_7_layer_call_and_return_conditional_losses_71436
G__inference_sequential_7_layer_call_and_return_conditional_losses_71106
G__inference_sequential_7_layer_call_and_return_conditional_losses_71134?
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
?2?
,__inference_sequential_7_layer_call_fn_71188
,__inference_sequential_7_layer_call_fn_71486
,__inference_sequential_7_layer_call_fn_71461
,__inference_sequential_7_layer_call_fn_71241?
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
 __inference__wrapped_model_70521?
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
annotations? */?,
*?'
lstm_4_input?????????`
?2?
A__inference_lstm_4_layer_call_and_return_conditional_losses_71554
A__inference_lstm_4_layer_call_and_return_conditional_losses_71520
A__inference_lstm_4_layer_call_and_return_conditional_losses_71644
A__inference_lstm_4_layer_call_and_return_conditional_losses_71610?
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
?2?
&__inference_lstm_4_layer_call_fn_71565
&__inference_lstm_4_layer_call_fn_71666
&__inference_lstm_4_layer_call_fn_71576
&__inference_lstm_4_layer_call_fn_71655?
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
B__inference_dense_8_layer_call_and_return_conditional_losses_71677?
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
'__inference_dense_8_layer_call_fn_71686?
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
?2?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71758
A__inference_lstm_5_layer_call_and_return_conditional_losses_71816
A__inference_lstm_5_layer_call_and_return_conditional_losses_71722
A__inference_lstm_5_layer_call_and_return_conditional_losses_71852?
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
?2?
&__inference_lstm_5_layer_call_fn_71780
&__inference_lstm_5_layer_call_fn_71874
&__inference_lstm_5_layer_call_fn_71769
&__inference_lstm_5_layer_call_fn_71863?
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
B__inference_dense_9_layer_call_and_return_conditional_losses_71885?
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
'__inference_dense_9_layer_call_fn_71894?
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
#__inference_signature_wrapper_71276lstm_4_input"?
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
?2??
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
?2??
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
?2??
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
?2??
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
 __inference__wrapped_model_70521z
()*+,-9?6
/?,
*?'
lstm_4_input?????????`
? "1?.
,
dense_9!?
dense_9?????????0?
B__inference_dense_8_layer_call_and_return_conditional_losses_71677e4?1
*?'
%?"
inputs?????????`?
? ")?&
?
0?????????`
? ?
'__inference_dense_8_layer_call_fn_71686X4?1
*?'
%?"
inputs?????????`?
? "??????????`?
B__inference_dense_9_layer_call_and_return_conditional_losses_71885\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????0
? z
'__inference_dense_9_layer_call_fn_71894O/?,
%?"
 ?
inputs?????????2
? "??????????0?
A__inference_lstm_4_layer_call_and_return_conditional_losses_71520?()*O?L
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
A__inference_lstm_4_layer_call_and_return_conditional_losses_71554?()*O?L
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
A__inference_lstm_4_layer_call_and_return_conditional_losses_71610r()*??<
5?2
$?!
inputs?????????`

 
p

 
? "*?'
 ?
0?????????`?
? ?
A__inference_lstm_4_layer_call_and_return_conditional_losses_71644r()*??<
5?2
$?!
inputs?????????`

 
p 

 
? "*?'
 ?
0?????????`?
? ?
&__inference_lstm_4_layer_call_fn_71565~()*O?L
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
&__inference_lstm_4_layer_call_fn_71576~()*O?L
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
&__inference_lstm_4_layer_call_fn_71655e()*??<
5?2
$?!
inputs?????????`

 
p

 
? "??????????`??
&__inference_lstm_4_layer_call_fn_71666e()*??<
5?2
$?!
inputs?????????`

 
p 

 
? "??????????`??
A__inference_lstm_5_layer_call_and_return_conditional_losses_71722m+,-??<
5?2
$?!
inputs?????????`

 
p

 
? "%?"
?
0?????????2
? ?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71758m+,-??<
5?2
$?!
inputs?????????`

 
p 

 
? "%?"
?
0?????????2
? ?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71816}+,-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????2
? ?
A__inference_lstm_5_layer_call_and_return_conditional_losses_71852}+,-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????2
? ?
&__inference_lstm_5_layer_call_fn_71769`+,-??<
5?2
$?!
inputs?????????`

 
p

 
? "??????????2?
&__inference_lstm_5_layer_call_fn_71780`+,-??<
5?2
$?!
inputs?????????`

 
p 

 
? "??????????2?
&__inference_lstm_5_layer_call_fn_71863p+,-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????2?
&__inference_lstm_5_layer_call_fn_71874p+,-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????2?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71106v
()*+,-A?>
7?4
*?'
lstm_4_input?????????`
p

 
? "%?"
?
0?????????0
? ?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71134v
()*+,-A?>
7?4
*?'
lstm_4_input?????????`
p 

 
? "%?"
?
0?????????0
? ?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71356p
()*+,-;?8
1?.
$?!
inputs?????????`
p

 
? "%?"
?
0?????????0
? ?
G__inference_sequential_7_layer_call_and_return_conditional_losses_71436p
()*+,-;?8
1?.
$?!
inputs?????????`
p 

 
? "%?"
?
0?????????0
? ?
,__inference_sequential_7_layer_call_fn_71188i
()*+,-A?>
7?4
*?'
lstm_4_input?????????`
p

 
? "??????????0?
,__inference_sequential_7_layer_call_fn_71241i
()*+,-A?>
7?4
*?'
lstm_4_input?????????`
p 

 
? "??????????0?
,__inference_sequential_7_layer_call_fn_71461c
()*+,-;?8
1?.
$?!
inputs?????????`
p

 
? "??????????0?
,__inference_sequential_7_layer_call_fn_71486c
()*+,-;?8
1?.
$?!
inputs?????????`
p 

 
? "??????????0?
#__inference_signature_wrapper_71276?
()*+,-I?F
? 
??<
:
lstm_4_input*?'
lstm_4_input?????????`"1?.
,
dense_9!?
dense_9?????????0