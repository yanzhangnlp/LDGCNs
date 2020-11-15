f1 = open('train.amr', 'r')
old_indexs = []
old_nodes_indexs = []
old_nodes = []
old_edges = []
for line in f1:
	seg = line.strip().split()
	map_indexs = {}
	key = 0
	nodes_indexs = []
	edges_indexs = []
	nodes = []
	edges = []
	for j in range(len(seg)):
		if seg[j][0] != ':':
			nodes_indexs.append(j)
			nodes.append(seg[j])
			map_indexs[j] = key 
			key += 1
		else:
			edges.append(seg[j])
			edges_indexs.append(j)

	if key != len(nodes_indexs):
		print("error:", seg)

	for t in range(len(edges_indexs)):
		map_indexs[edges_indexs[t]] = key
		key += 1
	old_indexs.append(map_indexs)
	old_nodes_indexs.append(nodes_indexs)
	old_edges.append(edges)
	old_nodes.append(nodes)


grh_triples = []
grh_id = 0
f2 = open('train.grh', 'r')
for line in f2:
	seg = line.strip().split()
	indexs_maps = old_indexs[grh_id]
	grh_id += 1
	sample = []
	for i in range(len(seg)):
		triple = seg[i].split(',')
		h = int(triple[0][1:])
		t = int(triple[1])
		r = seg[i][-2]
		map_triple = [indexs_maps[h], indexs_maps[t], r]
		sample.append(map_triple)
	grh_triples.append(sample)


new_nodes_index = []
bpw_new_nodes = []
amr_id = 0
f3 = open('train_nodes_bpe.amr', 'r')
for line in f3:
	seg = line.strip().split()
	nodes = old_nodes[amr_id]
	edges = old_edges[amr_id]
	amr_id += 1
	map_index = {}
	key = 0
	bpe_index = []
	new_nodes = []
	for i in range(len(seg)):
		w = seg[i]
		new_nodes.append(w)
		if nodes[key] == w:
			map_index[key] = i
			key += 1
		else:
			if w[-2:] == '@@':
				bpe_index.append(i)
			else:
				bpe_index.append(i)
				map_index[key] = bpe_index
				key += 1
				bpe_index = []
	if key != len(nodes):
		print("error")
	for j in range(len(edges)):
		map_index[key] = j + len(seg)
		key += 1
	new_nodes_index.append(map_index)
	bpw_new_nodes.append(new_nodes)

new_grh_triples = []
for i in range(len(grh_triples)):
	grh = grh_triples[i]
	map_index = new_nodes_index[i]
	new_sample = {}
	for j in range(len(grh)):
		triple = grh[j]
		h = triple[0]
		t = triple[1]
		r = triple[2]
		map_h = map_index[h]
		map_t = map_index[t]
		if isinstance(map_h, int) and isinstance(map_t, int):
			new_sample[tuple([map_h, map_t, r])] = 1
		elif isinstance(map_h, list) and isinstance(map_t, int):
			new_sample[tuple([map_h[-1], map_t, r])] = 1
			for k in range(len(map_h)):
				if k < len(map_h) -1:
					new_sample[tuple([map_h[k], map_h[k+1], 'd'])] = 1
					new_sample[tuple([map_h[k+1], map_h[k], 'r'])] = 1
					new_sample[tuple([map_h[k], map_h[k], 's'])] = 1

		elif isinstance(map_h, int) and isinstance(map_t, list):
			new_sample[tuple([map_h, map_t[0], r])] = 1
			for k in range(len(map_t)):
				if k < len(map_t) -1:
					new_sample[tuple([map_t[k], map_t[k+1], 'd'])] = 1
					new_sample[tuple([map_t[k+1], map_t[k], 'r'])] = 1
					new_sample[tuple([map_t[k], map_t[k], 's'])] =1

		elif isinstance(map_h, list) and isinstance(map_t, list):
			new_sample[tuple([map_h[-1], map_t[0], r])] = 1
			for k in range(len(map_h)):
				if k < len(map_h) -1:
					new_sample[tuple([map_h[k], map_h[k+1], 'd'])] = 1
					new_sample[tuple([map_h[k+1], map_h[k], 'r'])] = 1
					new_sample[tuple([map_h[k], map_h[k], 's'])] = 1
			for k in range(len(map_t)):
				if k < len(map_t) -1:
					new_sample[tuple([map_t[k], map_t[k+1], 'd'])] = 1
					new_sample[tuple([map_t[k+1], map_t[k], 'r'])] = 1
					new_sample[tuple([map_t[k], map_t[k], 's'])] = 1

	new_grh_triples.append(new_sample.keys())

print(len(new_grh_triples))
h1 = open('train_bpe.grh', 'w')
h2 = open('train_bpe.amr', 'w')
for i in range(len(new_grh_triples)):
	grh = new_grh_triples[i]
	new_nodes = bpw_new_nodes[i]
	edges = old_edges[i]
	for tup in grh:
		h1.write('(' + str(tup[0]) + ',' + str(tup[1]) + ',' + str(tup[2]) + ')' + ' ')
	h1.write('\n')
	for d in new_nodes:
		h2.write(str(d) + ' ')
	for e in edges:
		h2.write(str(e) + ' ')
	h2.write('\n')

		

# grh_triples = []
# f1 = open('dev_bpe.grh', 'r')
# for line in f1:
# 	seg = line.strip().split()
# 	grh_triples.append(seg)
		

# amr_nodes = []
# f2 = open('dev_bpe.amr', 'r')
# for line in f2:
# 	seg = line.strip().split()
# 	amr_nodes.append(seg)



# h = open('dev_bpe.amrgrh', 'w')

# for i in range(len(grh_triples)):
# 	grh = grh_triples[i]
# 	amr = amr_nodes[i]
# 	for n in amr:
# 		h.write(n + ' ')
# 	h.write('\t')
# 	for t in grh:
# 		h.write(t + ' ')
# 	h.write('\n')






















