"?9
uHostFlushSummaryWriter"FlushSummaryWriter(1?"??~??@9?"??~??@A?"??~??@I?"??~??@a?F?a`q??i?F?a`q???Unknown?
BHostIDLE"IDLE1NbX???@ANbX???@a?LoPf??i<??h?k???Unknown
sHost_FusedMatMul"sequential_1/dense_1/Relu(1????Kwm@9????Kwm@A????Kwm@I????Kwm@a??~????i4????o???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_1/MatMul(1??"??h@9??"??h@A??"??h@I??"??h@a?\
?OG??i?*??D???Unknown
^HostGatherV2"GatherV2(1?Zd?T@9?Zd?T@A?Zd?T@I?Zd?T@a????????i??X<????Unknown
`HostGatherV2"
GatherV2_1(1?rh??\P@9?rh??\P@A?rh??\P@I?rh??\P@a鴎g???iշ??X???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1?~j?t?F@9?~j?t?F@A?~j?t?F@I?~j?t?F@a???њƈ?i??@ ?????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1d;?O??@@9d;?O??@@Ad;?O??@@Id;?O??@@a????n??i??;lT???Unknown
?	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1bX9??@9bX9??@A)\???(;@I)\???(;@ah?c?}?i?D;3?@???Unknown
?
HostBiasAddGrad"6gradient_tape/sequential_1/dense_1/BiasAdd/BiasAddGrad(1?n???5@9?n???5@A?n???5@I?n???5@a?jq?w?i?hzp???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1?Zd?4@9?Zd?4@A?Zd?4@I?Zd?4@a??A??v?i|@?g????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_2/MatMul_1(133333?4@933333?4@A33333?4@I33333?4@a?d3??v?iE?",B????Unknown
vHost_FusedMatMul"sequential_1/dense_2/BiasAdd(1y?&11@9y?&11@Ay?&11@Iy?&11@a?3?B
?r?i?a?@?????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1j?t??0@9j?t??0@Aj?t??0@Ij?t??0@a
?/?)r?i??Ξ????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1+??η+@9+??η+@A+??η+@I+??η+@a=???Un?i??W;4???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(19??v??*@99??v??*@A9??v??*@I9??v??*@a*E8?7#m?iTɷ?^Q???Unknown
iHostWriteSummary"WriteSummary(1'1??)@9'1??)@A'1??)@I'1??)@a?×??l?iadrcm???Unknown?
VHostSum"Sum_2(1
ףp=
'@9
ףp=
'@A
ףp=
'@I
ףp=
'@aQ?u?7i?i????????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1????x?&@9????x?&@A????x?&@I????x?&@a7?? ;i?i??9ŭ????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1?"??~?$@9?"??~?$@A?"??~?$@I?"??~?$@a????f?iԾȑ????Unknown
vHostCast"$sparse_categorical_crossentropy/Cast(1`??"??!@9`??"??!@A`??"??!@I`??"??!@a?LT)Q?c?i?(?>????Unknown
dHostDataset"Iterator::Model(1?|?5^6@9?|?5^6@A?E????!@I?E????!@a_U3?=c?i?}|????Unknown
?HostReluGrad"+gradient_tape/sequential_1/dense_1/ReluGrad(1?$??!@9?$??!@A?$??!@I?$??!@a[??d˞b?i?T??????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_4(1??|?5?@9??|?5?@A??|?5?@I??|?5?@a?@aH6pa?iܵ?	????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1Zd;??@9Zd;??@AZd;??@IZd;??@a??^ha?i???g????Unknown
ZHostArgMax"ArgMax(1B`??"?@9B`??"?@AB`??"?@IB`??"?@a?y0?\V`?i??I#???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1???K7?2@9???K7?2@AR????@IR????@a?r?QQ`?i?v.?3???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1??ʡ?L@9??ʡ?L@AL7?A`e@IL7?A`e@a<?p?_?i?}??$C???Unknown
?HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1-?????@9-?????@A-?????@I-?????@a'?LZ?iT???'P???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1{?G?z@9{?G?z@A{?G?z@I{?G?z@a;v?]?Y?i_}?? ]???Unknown
?HostReadVariableOp"+sequential_1/dense_1/BiasAdd/ReadVariableOp(1?Q??k@9?Q??k@A?Q??k@I?Q??k@a5w~?YV?i?<EH-h???Unknown
l HostIteratorGetNext"IteratorGetNext(1?Zd;@9?Zd;@A?Zd;@I?Zd;@a???`$V?is??x?s???Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1???S??@9???S??@A???S??@I???S??@a??oU?iJ????}???Unknown
e"Host
LogicalAnd"
LogicalAnd(1?p=
ף@9?p=
ף@A?p=
ף@I?p=
ף@a6*&.NS?iQ????????Unknown?
?#HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_2(1??????@9??????@A??????@I??????@a?????*R?i>??W?????Unknown
?$HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ˡE???@9ˡE?????AˡE???@IˡE?????aB???P?i?u?7????Unknown
X%HostEqual"Equal(1?V-@9?V-@A?V-@I?V-@aB?Ԓ5?O?i?*?2????Unknown
?&HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1??Q??@9??Q??@A??Q??@I??Q??@a???k6O?i???? ????Unknown
t'HostAssignAddVariableOp"AssignAddVariableOp(1??x?&1@9??x?&1@A??x?&1@I??x?&1@a.???Z?M?i[??p????Unknown
b(HostDivNoNan"div_no_nan_1(1+????	@9+????	@A+????	@I+????	@a?SbmkL?i;?ⱋ????Unknown
?)HostReadVariableOp"*sequential_1/dense_2/MatMul/ReadVariableOp(1??ʡE@9??ʡE@A??ʡE@I??ʡE@a???xI?i??·?????Unknown
|*HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@aɤ?y?H?i?R
?????Unknown
?+HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1???S??@9???S??@A???S??@I???S??@aT?0D?H?i??'Q????Unknown
?,HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1?Q???@9?Q???@A?Q???@I?Q???@a2??i"?F?i??o
????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_3(1T㥛? @9T㥛? @AT㥛? @IT㥛? @a?v?=F?i?-??????Unknown
X.HostCast"Cast_3(1)\???(@9)\???(@A)\???(@I)\???(@a?&Y=??C?iD??6?????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_1(1ˡE??? @9ˡE??? @AˡE??? @IˡE??? @a?$}{??B?i?c_*????Unknown
w0HostReadVariableOp"div_no_nan_1/ReadVariableOp(1w??/???9w??/???Aw??/???Iw??/???a???5??@?i??RCc????Unknown
T1HostMul"Mul(1??MbX??9??MbX??A??MbX??I??MbX??a
?4??@?i?l5?f????Unknown
s2HostReadVariableOp"SGD/Cast/ReadVariableOp(1?????M??9?????M??A?????M??I?????M??ai?}e??<?irB ????Unknown
?3HostReadVariableOp"*sequential_1/dense_1/MatMul/ReadVariableOp(1+??????9+??????A+??????I+??????a?Sbmk<?i?f????Unknown
?4HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1???S???9???S???A???S???I???S???aV?$?U<?i~+_,????Unknown
y5HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(11?Zd??91?Zd??A1?Zd??I1?Zd??a?Ӂ?_i7?i?[XX????Unknown
u6HostReadVariableOp"div_no_nan/ReadVariableOp(1?ʡE????9?ʡE????A?ʡE????I?ʡE????a?y:N??5?i#"?????Unknown
u7HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1?n?????9?n?????A?n?????I?n?????a&sO?x3?i?̤./????Unknown
`8HostDivNoNan"
div_no_nan(1333333??9333333??A333333??I333333??aM?)???2?i(2???????Unknown
w9HostReadVariableOp"div_no_nan/ReadVariableOp_1(1=
ףp=??9=
ףp=??A=
ףp=??I=
ףp=??a?׾??,?i??U????Unknown
a:HostIdentity"Identity(1????Mb??9????Mb??A????Mb??I????Mb??a?????*?i?????????Unknown?*?8
uHostFlushSummaryWriter"FlushSummaryWriter(1?"??~??@9?"??~??@A?"??~??@I?"??~??@a??ۀ???i??ۀ????Unknown?
sHost_FusedMatMul"sequential_1/dense_1/Relu(1????Kwm@9????Kwm@A????Kwm@I????Kwm@aZޯ<9??i??1?#???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_1/MatMul(1??"??h@9??"??h@A??"??h@I??"??h@a??W???i+?*?u????Unknown
^HostGatherV2"GatherV2(1?Zd?T@9?Zd?T@A?Zd?T@I?Zd?T@a??????i??U????Unknown
`HostGatherV2"
GatherV2_1(1?rh??\P@9?rh??\P@A?rh??\P@I?rh??\P@a?]iP???iI?4?????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1?~j?t?F@9?~j?t?F@A?~j?t?F@I?~j?t?F@a|B"G\???i]?m?r???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1d;?O??@@9d;?O??@@Ad;?O??@@Id;?O??@@a????????i[??6????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1bX9??@9bX9??@A)\???(;@I)\???(;@a??B??S??i?f?U?????Unknown
?	HostBiasAddGrad"6gradient_tape/sequential_1/dense_1/BiasAdd/BiasAddGrad(1?n???5@9?n???5@A?n???5@I?n???5@ap??6???i	??-/'???Unknown
}
HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1?Zd?4@9?Zd?4@A?Zd?4@I?Zd?4@a+s?[????i?^h'.l???Unknown
HostMatMul"+gradient_tape/sequential_1/dense_2/MatMul_1(133333?4@933333?4@A33333?4@I33333?4@av??Q9??ihޯ0????Unknown
vHost_FusedMatMul"sequential_1/dense_2/BiasAdd(1y?&11@9y?&11@Ay?&11@Iy?&11@aX?p7|?i??Ȟ ????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1j?t??0@9j?t??0@Aj?t??0@Ij?t??0@a?v?bI{?i??8d????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1+??η+@9+??η+@A+??η+@I+??η+@a?J3k\?v?i'cFM???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(19??v??*@99??v??*@A9??v??*@I9??v??*@a0???u?i???\y???Unknown
iHostWriteSummary"WriteSummary(1'1??)@9'1??)@A'1??)@I'1??)@a|??b
u?i\x?q$????Unknown?
VHostSum"Sum_2(1
ףp=
'@9
ףp=
'@A
ףp=
'@I
ףp=
'@a]!???r?i??(????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1????x?&@9????x?&@A????x?&@I????x?&@a."?j??r?iZ????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1?"??~?$@9?"??~?$@A?"??~?$@I?"??~?$@a?|u??1q?iT|?????Unknown
vHostCast"$sparse_categorical_crossentropy/Cast(1`??"??!@9`??"??!@A`??"??!@I`??"??!@aD˟=?m?i???.???Unknown
dHostDataset"Iterator::Model(1?|?5^6@9?|?5^6@A?E????!@I?E????!@a?9?f?l?iV0X?K???Unknown
?HostReluGrad"+gradient_tape/sequential_1/dense_1/ReluGrad(1?$??!@9?$??!@A?$??!@I?$??!@a?88S?k?i??h??g???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_4(1??|?5?@9??|?5?@A??|?5?@I??|?5?@aJ????2j?i=j?????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1Zd;??@9Zd;??@AZd;??@IZd;??@a??]??&j?i
r?_ߛ???Unknown
ZHostArgMax"ArgMax(1B`??"?@9B`??"?@AB`??"?@IB`??"?@a?)?O?h?i$?V?j????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1???K7?2@9???K7?2@AR????@IR????@a?!/??h?i???k?????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1??ʡ?L@9??ʡ?L@AL7?A`e@IL7?A`e@a???zXg?i?? xF????Unknown
?HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1-?????@9-?????@A-?????@I-?????@aE/?s??c?i?<t8?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1{?G?z@9{?G?z@A{?G?z@I{?G?z@a;??uMc?i~(?????Unknown
?HostReadVariableOp"+sequential_1/dense_1/BiasAdd/ReadVariableOp(1?Q??k@9?Q??k@A?Q??k@I?Q??k@a?X??w?`?i׫|%????Unknown
lHostIteratorGetNext"IteratorGetNext(1?Zd;@9?Zd;@A?Zd;@I?Zd;@ay??5??`?i????,???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1???S??@9???S??@A???S??@I???S??@aF??&`?iϛ5?<???Unknown
e!Host
LogicalAnd"
LogicalAnd(1?p=
ף@9?p=
ף@A?p=
ף@I?p=
ף@al?7? ]?iZ}Qt2K???Unknown?
?"HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_2(1??????@9??????@A??????@I??????@axڑ?K[?iG?.??X???Unknown
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ˡE???@9ˡE?????AˡE???@IˡE?????a?c*czY?iy۹.?e???Unknown
X$HostEqual"Equal(1?V-@9?V-@A?V-@I?V-@a??3?W?i?ǮH?q???Unknown
?%HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1??Q??@9??Q??@A??Q??@I??Q??@a?? %rW?i?9?[L}???Unknown
t&HostAssignAddVariableOp"AssignAddVariableOp(1??x?&1@9??x?&1@A??x?&1@I??x?&1@a=q??ZV?i?A?y????Unknown
b'HostDivNoNan"div_no_nan_1(1+????	@9+????	@A+????	@I+????	@a??YU?i?΍8&????Unknown
?(HostReadVariableOp"*sequential_1/dense_2/MatMul/ReadVariableOp(1??ʡE@9??ʡE@A??ʡE@I??ʡE@a??3ٮ!S?i?h??????Unknown
|)HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@a;;L&ǦR?i???s
????Unknown
?*HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1???S??@9???S??@A???S??@I???S??@a?H\?R?iB???Y????Unknown
?+HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1?Q???@9?Q???@A?Q???@I?Q???@a?.??2Q?i?????????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_3(1T㥛? @9T㥛? @AT㥛? @IT㥛? @a???P?i????8????Unknown
X-HostCast"Cast_3(1)\???(@9)\???(@A)\???(@I)\???(@ae?[v??M?i?????????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_1(1ˡE??? @9ˡE??? @AˡE??? @IˡE??? @a????K?i:???????Unknown
w/HostReadVariableOp"div_no_nan_1/ReadVariableOp(1w??/???9w??/???Aw??/???Iw??/???a?(??r_I?i"?m?????Unknown
T0HostMul"Mul(1??MbX??9??MbX??A??MbX??I??MbX??aH.P?H?i.K5?????Unknown
s1HostReadVariableOp"SGD/Cast/ReadVariableOp(1?????M??9?????M??A?????M??I?????M??a9zǟE?i<??s????Unknown
?2HostReadVariableOp"*sequential_1/dense_1/MatMul/ReadVariableOp(1+??????9+??????A+??????I+??????a??YE?i??X??????Unknown
?3HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1???S???9???S???A???S???I???S???a????:HE?i?9?????Unknown
y4HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(11?Zd??91?Zd??A1?Zd??I1?Zd??a?'G??A?i|??s?????Unknown
u5HostReadVariableOp"div_no_nan/ReadVariableOp(1?ʡE????9?ʡE????A?ʡE????I?ʡE????a?%??f@?i??u2?????Unknown
u6HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1?n?????9?n?????A?n?????I?n?????a$????@=?i<1?KC????Unknown
`7HostDivNoNan"
div_no_nan(1333333??9333333??A333333??I333333??a?A.*?G<?iwm??????Unknown
w8HostReadVariableOp"div_no_nan/ReadVariableOp_1(1=
ףp=??9=
ףp=??A=
ףp=??I=
ףp=??a?mfO?5?i?DZ?~????Unknown
a9HostIdentity"Identity(1????Mb??9????Mb??A????Mb??I????Mb??a???-?4?i      ???Unknown?2CPU