## Build
<pre>
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
</pre>

## Run
<pre>
cd build/examples
./onetla
</pre>

## Show(A770)
<pre>
host: 23824ms
naive gpu: 86ms
naive by esimd gpu: 43ms
gpu: 120ms
</pre>