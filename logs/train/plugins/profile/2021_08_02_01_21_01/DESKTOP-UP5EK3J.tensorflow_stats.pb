"?9
BHostIDLE"IDLE1     ??@A     ??@a??(??*??i??(??*???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a?l&?????ir?'˥????Unknown?
sHost_FusedMatMul"sequential_1/dense_1/Relu(1      g@9      g@A      g@I      g@al?Nh?2??ii???ӓ???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_1/MatMul(133333?]@933333?]@A33333?]@I33333?]@a??s????i????̬???Unknown
^HostGatherV2"GatherV2(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a??t?Y#??i?p|??U???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(133333?:@933333?:@A33333?:@I33333?:@a??q???i?T??????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(13333338@93333338@A?????5@I?????5@aš??x?i`???????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1fffff?1@9fffff?1@Afffff?1@Ifffff?1@a?P?*u?iׁe?J????Unknown
}	HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1??????/@9??????/@A??????/@I??????/@a?O?.?r?iwR?????Unknown

HostMatMul"+gradient_tape/sequential_1/dense_2/MatMul_1(1??????-@9??????-@A??????-@I??????-@a+֋xr?q?i#j??9???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff(@9ffffff(@Affffff(@Iffffff(@a9 j???l?i#?j??V???Unknown
?HostReadVariableOp"*sequential_1/dense_1/MatMul/ReadVariableOp(1??????$@9??????$@A??????$@I??????$@at??:ؘh?iӊ??o???Unknown
iHostWriteSummary"WriteSummary(1??????$@9??????$@A??????$@I??????$@a?d?ML\h?i8b??y????Unknown?
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_1/BiasAdd/BiasAddGrad(1??????"@9??????"@A??????"@I??????"@a|q??e?i?? ?x????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1??????!@9??????!@A??????!@I??????!@a???l?d?i?Bm?H????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1ffffff!@9ffffff!@Affffff!@Iffffff!@a9????d?iM%?R?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????,@9??????,@A??????@I??????@a̸)?6b?iO,i????Unknown
dHostDataset"Iterator::Model(1     ?3@9     ?3@A333333@I333333@a?????Ca?i???OV????Unknown
vHost_FusedMatMul"sequential_1/dense_2/BiasAdd(1??????@9??????@A??????@I??????@au\/ضQ`?i?*??????Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a????=?_?i?????
???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a??"#&8_?i,-?????Unknown
vHostCast"$sparse_categorical_crossentropy/Cast(1333333@9333333@A333333@I333333@a?Gu?Y?i?g???&???Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@aT??x?V?i?E??1???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a?%? I?U?i?e??<???Unknown
uHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aڎ?F1IU?iԥG??G???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffffE@9ffffffE@A333333@I333333@a?`?WT?i?'+?Q???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a?`?WT?i4?ګ?[???Unknown
ZHostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a?2???dS?iMl??e???Unknown
XHostEqual"Equal(1      @9      @A      @I      @am????R?iP??o???Unknown
VHostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@aAmJR??Q?iRu??x???Unknown
?HostReluGrad"+gradient_tape/sequential_1/dense_1/ReluGrad(1ffffff@9ffffff@Affffff@Iffffff@aAmJR??Q?i??|?????Unknown
? HostReadVariableOp"*sequential_1/dense_2/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@aAmJR??Q?i??HA?????Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff
@9ffffff
@Affffff
@Iffffff
@a??"#&8O?i}??Jɑ???Unknown
e"Host
LogicalAnd"
LogicalAnd(1ffffff
@9ffffff
@Affffff
@Iffffff
@a??"#&8O?i:QZT?????Unknown?
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9????????A??????@I????????aO?(??SM?i`[	F?????Unknown
l$HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a#i??aL?i:?˫????Unknown
?%HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a#i??aL?i??????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a???7}J?iWv_?????Unknown
`'HostDivNoNan"
div_no_nan(1ffffff@9ffffff@Affffff@Iffffff@a???7}J?i??^?[????Unknown
?(HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1??????@9??????@A??????@I??????@at??:ؘH?iF7m??????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?%? I?E?iG?u?????Unknown
s*HostReadVariableOp"SGD/Cast/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?%? I?E?i?V?c????Unknown
?+HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1ffffff@9ffffff@Affffff@Iffffff@a?%? I?E?i?fE??????Unknown
?,HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff@9ffffff@Affffff@Iffffff@a?%? I?E?ijv?,D????Unknown
?-HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a???l?D?i???2x????Unknown
u.HostReadVariableOp"div_no_nan/ReadVariableOp(1?????? @9?????? @A?????? @I?????? @a??D???C?i?V?o????Unknown
|/HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1????????9????????A????????I????????a{ťo?E>?i??$l8????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a???7}:?i?"?????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff??9ffffff??Affffff??Iffffff??a???7}:?iY??????Unknown
?2HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_2(1????????9????????A????????I????????at??:ؘ8?i????????Unknown
?3HostReadVariableOp"+sequential_1/dense_1/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??aT??x?6?ix'/d?????Unknown
T4HostMul"Mul(1????????9????????A????????I????????a???l?4?i??\g[????Unknown
X5HostCast"Cast_3(1      ??9      ??A      ??I      ??am????2?i?x?޸????Unknown
b6HostDivNoNan"div_no_nan_1(1????????9????????A????????I????????a?͞Z1?iRR???????Unknown
w7HostReadVariableOp"div_no_nan/ReadVariableOp_1(1????????9????????A????????I????????a{ťo?E.?i?LX)?????Unknown
a8HostIdentity"Identity(1333333??9333333??A333333??I333333??aT??x?&?is??p)????Unknown?
w9HostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333??9333333??A333333??I333333??aT??x?&?i8?r??????Unknown
y:HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333??9333333??A333333??I333333??aT??x?&?i?????????Unknown*?8
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a??? I???i??? I????Unknown?
sHost_FusedMatMul"sequential_1/dense_1/Relu(1      g@9      g@A      g@I      g@aB??&x???i?(??????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_1/MatMul(133333?]@933333?]@A33333?]@I33333?]@a?m ??!??i56
?????Unknown
^HostGatherV2"GatherV2(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a?J+k??iVw?X????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(133333?:@933333?:@A33333?:@I33333?:@a???qw??i?{6͎???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(13333338@93333338@A?????5@I?????5@adwں???infy}????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1fffff?1@9fffff?1@Afffff?1@Ifffff?1@a#j?-r??i9?E8???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1??????/@9??????/@A??????/@I??????/@aq?	?*??i_?X?|???Unknown
	HostMatMul"+gradient_tape/sequential_1/dense_2/MatMul_1(1??????-@9??????-@A??????-@I??????-@aL??,???iuM>????Unknown
s
HostDataset"Iterator::Model::ParallelMapV2(1ffffff(@9ffffff(@Affffff(@Iffffff(@a?o?сz?i???A????Unknown
?HostReadVariableOp"*sequential_1/dense_1/MatMul/ReadVariableOp(1??????$@9??????$@A??????$@I??????$@a&V???v?i?*I5s???Unknown
iHostWriteSummary"WriteSummary(1??????$@9??????$@A??????$@I??????$@aR???av?i??B;5L???Unknown?
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_1/BiasAdd/BiasAddGrad(1??????"@9??????"@A??????"@I??????"@a
??4t?i??KӞt???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1??????!@9??????!@A??????!@I??????!@a?Hr??s?i_?\4ܚ???Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1ffffff!@9ffffff!@Affffff!@Iffffff!@a??o?r?io<W?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????,@9??????,@A??????@I??????@a??cwںp?io?* ????Unknown
dHostDataset"Iterator::Model(1     ?3@9     ?3@A333333@I333333@a??1(??o?ih?R?????Unknown
vHost_FusedMatMul"sequential_1/dense_2/BiasAdd(1??????@9??????@A??????@I??????@aP??a??m?iZ???????Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a ?P?Fm?iI????<???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@aWm???l?i??۟Y???Unknown
vHostCast"$sparse_categorical_crossentropy/Cast(1333333@9333333@A333333@I333333@a??CG?g?i??ƻ?p???Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a?OO??d?i??e?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a5L??,?c?i*???????Unknown
uHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?????c?i????????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffffE@9ffffffE@A333333@I333333@a=??Vr?b?i????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a=??Vr?b?i?Riem????Unknown
ZHostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a?Á???a?iG?\[>????Unknown
XHostEqual"Equal(1      @9      @A      @I      @aDB???aa?i???????Unknown
VHostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@a?>?^;?`?i?A}N#???Unknown
?HostReluGrad"+gradient_tape/sequential_1/dense_1/ReluGrad(1ffffff@9ffffff@Affffff@Iffffff@a?>?^;?`?i?ۉ????Unknown
?HostReadVariableOp"*sequential_1/dense_2/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a?>?^;?`?iFd:?)'???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff
@9ffffff
@Affffff
@Iffffff
@aWm???\?i???ɀ5???Unknown
e!Host
LogicalAnd"
LogicalAnd(1ffffff
@9ffffff
@Affffff
@Iffffff
@aWm???\?i???C???Unknown?
?"HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9????????A??????@I????????a?f?Z?ig
VPQ???Unknown
l#HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @agcʢ?Z?i?۟Y^???Unknown
?$HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @agcʢ?Z?i????bk???Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a?\4ܚUX?i????w???Unknown
`&HostDivNoNan"
div_no_nan(1ffffff@9ffffff@Affffff@Iffffff@a?\4ܚUX?i'???????Unknown
?'HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1??????@9??????@A??????@I??????@a&V???V?iR???????Unknown
t(HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a5L??,?S?i??	l????Unknown
s)HostReadVariableOp"SGD/Cast/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a5L??,?S?i??????Unknown
?*HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1ffffff@9ffffff@Affffff@Iffffff@a5L??,?S?iD??? ????Unknown
?+HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff@9ffffff@Affffff@Iffffff@a5L??,?S?i?bk/?????Unknown
?,HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a?Hr??S?i????????Unknown
u-HostReadVariableOp"div_no_nan/ReadVariableOp(1?????? @9?????? @A?????? @I?????? @a?E'%4@R?i?/¡?????Unknown
|.HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1????????9????????A????????I????????aj`i??K?i̇܄?????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a?\4ܚUH?i㔓??????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff??9ffffff??Affffff??Iffffff??a?\4ܚUH?i??JR?????Unknown
?1HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_2(1????????9????????A????????I????????a&V???F?i?	?zs????Unknown
?2HostReadVariableOp"+sequential_1/dense_1/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??a?OO??D?i??#e?????Unknown
T3HostMul"Mul(1????????9????????A????????I????????a?Hr??C?i6?Er????Unknown
X4HostCast"Cast_3(1      ??9      ??A      ??I      ??aDB???aA?iG_6?????Unknown
b5HostDivNoNan"div_no_nan_1(1????????9????????A????????I????????aHw??}I??i?0???????Unknown
w6HostReadVariableOp"div_no_nan/ReadVariableOp_1(1????????9????????A????????I????????aj`i??;?i?\??-????Unknown
a7HostIdentity"Identity(1333333??9333333??A333333??I333333??a?OO??4?i?=??????Unknown?
w8HostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333??9333333??A333333??I333333??a?OO??4?i?֊d????Unknown
y9HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333??9333333??A333333??I333333??a?OO??4?i      ???Unknown2CPU