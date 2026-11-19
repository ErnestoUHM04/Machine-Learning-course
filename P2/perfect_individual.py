# ML P2
# El individuo perfecto
# Ernesto Ulises Hernández Martínez
import random

peso = float(input("mete tu peso (kg): "))
altura = float(input("mete tu altura (m): "))

imc = peso / altura**2

print(f"Tu IMC es: {imc:.2f}")
