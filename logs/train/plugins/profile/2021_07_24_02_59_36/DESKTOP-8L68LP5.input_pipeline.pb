	2???z@2???z@!2???z@	??,?a|3@??,?a|3@!??,?a|3@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:2???z@?:TS?u??A????@Y?.?.G??rEagerKernelExecute 0*	P??n?{@2F
Iterator::Model???,z??!??I-t?L@)RH2?w???1UѾ?{?I@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceF
e??k??!???zBx.@)F
e??k??1???zBx.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip~!<?8??!L?ҋ.E@)<?Bus???1t?SS?$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?w??#???!????39@)??n?o???1??.%?#@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?l\???!?\??X@)?N\?W ??1?????@:Preprocessing2U
Iterator::Model::ParallelMapV2?
?rߙ?!??X\ğ@)?
?rߙ?1??X\ğ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorsI?v|s?!??A?	??)sI?v|s?1??A?	??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 19.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s3.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??,?a|3@IEȴ?? T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:TS?u???:TS?u??!?:TS?u??      ??!       "      ??!       *      ??!       2	????@????@!????@:      ??!       B      ??!       J	?.?.G???.?.G??!?.?.G??R      ??!       Z	?.?.G???.?.G??!?.?.G??b      ??!       JCPU_ONLYY??,?a|3@b qEȴ?? T@