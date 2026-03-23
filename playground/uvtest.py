from quantenschaltungengpus import hello
from quantenschaltungengpus.paulimatrizen import paulix, pauliy, pauliz


print(hello())
# print(paulix())
# print(pauliy())
# print(pauliz())

print(f"{paulix() @ pauliy()}={1j * pauliz()}")


# hello_str=hello()

# hello_str.capitalize
# hello_str.find
