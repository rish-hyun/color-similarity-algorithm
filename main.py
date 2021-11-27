import requirements
from color import Color

COLOR_LIST = [
    {'name': 'BLACK', 'r': 8.6, 'g': 6.45, 'b': 5.25},
    {'name': 'GRAY', 'r': 97.45, 'g': 93.75, 'b': 105.95},
    {'name': 'WHITE', 'r': 249.4, 'g': 246.5, 'b': 237.35},
    {'name': 'DARK_RED', 'r': 102.0, 'g': 14.5, 'b': 10.3},
    {'name': 'LIGHT_RED', 'r': 174.3, 'g': 32.4, 'b': 23.0},
    {'name': 'DARK_ORANGE', 'r': 173.1, 'g': 76.6, 'b': 7.9},
    {'name': 'LIGHT_ORANGE', 'r': 234.9, 'g': 129.3, 'b': 33.0},
    {'name': 'DARK_YELLOW', 'r': 214.6, 'g': 185.7, 'b': 111.3},
    {'name': 'LIGHT_YELLOW', 'r': 242.4, 'g': 227.3, 'b': 158.9},
    {'name': 'DARK_GREEN', 'r': 57.0, 'g': 114.2, 'b': 40.5},
    {'name': 'LIGHT_GREEN', 'r': 116.6, 'g': 210.9, 'b': 107.1},
    {'name': 'DARK_BLUE', 'r': 23.8, 'g': 35.9, 'b': 120.9},
    {'name': 'LIGHT_BLUE', 'r': 68.2, 'g': 140.6, 'b': 170.5},
    {'name': 'DARK_PURPLE', 'r': 85.9, 'g': 8.7, 'b': 56.4},
    {'name': 'LIGHT_PURPLE', 'r': 163.5, 'g': 99.1, 'b': 191.3},
    {'name': 'DARK_PINK', 'r': 238.7, 'g': 78.9, 'b': 128.4},
    {'name': 'LIGHT_PINK', 'r': 250.1, 'g': 163.9, 'b': 187.4},
    {'name': 'DARK_BROWN', 'r': 57.5, 'g': 32.1, 'b': 12.7},
    {'name': 'LIGHT_BROWN', 'r': 99.4, 'g': 62.4, 'b': 31.0}
]

if __name__ == '__main__':
    url_list = ['https://i.pinimg.com/564x/e4/a5/2d/e4a52d366b82cf6d414c437956edf196.jpg',
                'https://i.pinimg.com/564x/bd/66/47/bd664708cfc1f089a2ab9544252f2793.jpg',
                'https://i.pinimg.com/564x/78/09/ee/7809ee375a913f1968a4bf2608618674.jpg',
                'https://i.pinimg.com/564x/34/0f/1f/340f1f53931092d3a5ba6aba21e9743f.jpg',
                'https://i.pinimg.com/564x/56/5a/7f/565a7f4e176710c3860c40bdb7ec597b.jpg',
                'https://i.pinimg.com/564x/26/df/ea/26dfea24a42e600d8cccba856dc2bca8.jpg',
                'https://i.pinimg.com/564x/77/39/62/7739626e5943f21798e6281236c54788.jpg',
                'https://i.pinimg.com/564x/ce/3c/59/ce3c592a3cc5cbdf01188ff978b42de7.jpg',
                'https://i.pinimg.com/564x/10/8b/fb/108bfb9ac4f11ca988b87977bb627593.jpg',
                'https://i.pinimg.com/564x/ca/4e/b5/ca4eb5d83dcbf775a0155a32b7944e8d.jpg',
                'https://i.pinimg.com/564x/73/d2/ce/73d2ce9238857756ad7016e2a36fea93.jpg',
                'https://i.pinimg.com/564x/7e/a9/ab/7ea9ab41cdbb06d08f3193aa96c32828.jpg',
                'https://i.pinimg.com/564x/fe/18/a8/fe18a86f23f532dbef3918d3f7769d0a.jpg',
                'https://i.pinimg.com/564x/fd/1b/93/fd1b934a0ded71f40f247bbf3fb417d5.jpg',
                'https://i.pinimg.com/564x/33/5f/3b/335f3b6e31d32c101c0bf84d52498ad7.jpg',
                'https://i.pinimg.com/564x/22/59/c6/2259c6279433109a80aa1c8857623bf1.jpg',
                'https://i.pinimg.com/564x/e8/a4/8b/e8a48bb2bb1cb33f30e84549af1c388b.jpg',
                'https://i.pinimg.com/564x/b1/96/a9/b196a9d89db1890622f4358476ca6183.jpg',
                'https://i.pinimg.com/564x/94/c6/88/94c68812c2f3ac69f0f7942915c15569.jpg',
                'https://i.pinimg.com/564x/01/7d/e4/017de477cab592d65b7e6b0ea4811399.jpg',
                'https://i.pinimg.com/564x/ba/db/76/badb76105899c38b114e9bb35992717f.jpg',
                'https://i.pinimg.com/564x/d3/3f/f9/d33ff93062991ecc3952f09bf0fbbf3c.jpg',
                'https://i.pinimg.com/564x/b1/e6/fc/b1e6fc03c86512870630851ea09cfa85.jpg',
                'https://i.pinimg.com/564x/a1/43/20/a1432019772dbb5b2ef600cbf57c82cc.jpg']


    Color(None).cruv_method([10,100,200], COLOR_LIST, 'result.png')

    for url in url_list:
         color = Color(url)
         file_name = f"results/{url.split('/')[-1]}"
         color.get_match(COLOR_LIST, file_name)
