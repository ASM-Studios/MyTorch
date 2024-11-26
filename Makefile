all:	my_torch_analyzer my_torch_generator

my_torch_analyzer:
	cp my_torch_analyzer.py my_torch_analyzer
	chmod 755 my_torch_analyzer

my_torch_generator:
	cp my_torch_generator.py my_torch_generator
	chmod 755 my_torch_generator

clean:

fclean:
	rm -rf my_torch_analyzer
	rm -rf my_torch_generator

.PHONY:	all my_torch_analyzer my_torch_generator
