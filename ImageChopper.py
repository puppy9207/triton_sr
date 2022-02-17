import cv2
import numpy as np
# 자르기 현황을 보고 싶으면 True
debug_show = True
# 자른 범위 표시
is_rectangle = True
# 그림 저장 여부
is_save = False
# CHW?​
"""
    ImageChopper 사용법
    ----------------------
    (1) 클래스 선언
    var = ImageChopper(패딩 사이즈, 초해상화 배율)
    (2) 이미지 자르기 혹은 작은 이미지는 그대로 반환
    image_list = var.do_slicing_images(이미지) 
    잘려진 이미지에는 배치 사이즈가 없음
    (3) 추론
    (4) 이미지 합치기
    image = var.get_merge_image(image_list)
    합쳐진 이미지에는 배치 사이즈가 없음
"""
def batch_and_transpose(img, flag, batch_size=True):
    # 배치 사이즈가 처음부터 없다면 아래를 false로 바꿀 것
    is_batch_size = batch_size
    after_img = img
    if flag == 'CHW_TO_HWC':
        if is_batch_size:
            # REMOVE BATCH_SIZE
            after_img = np.squeeze(after_img)
        # CHW->HWC
        after_img = np.transpose(after_img, (1, 2, 0))
    elif flag == 'HWC_TO_CHW':
        # HWC->CHW
        after_img = np.transpose(after_img, (2, 0, 1))
        if is_batch_size:
            # ADD BATCH_SIZE
            after_img = np.expand_dims(after_img, axis=0)
    return after_img
def make_even_image(img_list):
    odd_count_list = []
    for idx, img in enumerate(img_list):
        odd_count = [0, 0]
        hh, ww, _ = img.shape
        if hh % 2 != 0:
            img_list[idx] = np.pad(img_list[idx], ((0, 1), (0, 0), (0, 0)), 'reflect')
            odd_count[0] = 1
        if ww % 2 != 0:
            img_list[idx] = np.pad(img_list[idx], ((0, 0), (0, 1), (0, 0)), 'reflect')
            odd_count[1] = 1
        odd_count_list.append(odd_count)
        print(odd_count)
    return img_list, odd_count_list

def _image_slicing(image, div_height_size, div_width_size, padding_size):
    # image : 이미지, div_height_size : 자를 높이 이미지 크기, div_width_size : 자를 이미지 넓이 크기
    # padding_size : div 크기에서 더 잘라 겹칠 크기

    # H W C
    original_height, original_width, _ = image.shape
    slicing_img_list = list()

    for row in range(0, original_height - 1, div_height_size):
        # 원래 넓이 사이즈에서 나눌 크기 만큼 반복
        cropped_img_list = list()
        for col in range(0, original_width - 1, div_width_size):
            # 원래 높이 사이즈에서 나눌 크기 만큼 반복
            # 해당 지점 x, y부터 나눌 크기 + 패딩 값 만큼 자르기
            after_row = row + div_height_size + padding_size
            if after_row > original_height:
                after_row = original_height

            after_col = col + div_width_size + padding_size
            if after_col > original_width:
                after_col = original_width

            #            if is_rectangle:
            #                image = cv2.rectangle(image, (col, row), (col + after_col, row + after_row), (0, 255, 0), 1)
            cropped_img = image[row: after_row, col: after_col]
            # debug_show_image(cropped_img)
            # 한줄 저장
            cropped_img_list.append(cropped_img)
        # 한줄씩 합쳐서 저장
        slicing_img_list.append(cropped_img_list)

    # 이미지 자르기 확인
    # debug_show_image(image)

    w_count = len(slicing_img_list)
    h_count = len(slicing_img_list[0])

    # 1차원 배열로 전환 (트리톤 메모리 전용)
    slicing_img_list = sum(slicing_img_list, [])
    for i in slicing_img_list:
        print(i.shape)
    slicing_img_list, odd_count_list = make_even_image(slicing_img_list)
    for i in slicing_img_list:
        print(i.shape)
    return slicing_img_list, h_count, w_count, odd_count_list


class ImageChopper:
    def __init__(self, padding_size, upscale_rate):
        self.padding_size = padding_size
        self.upscaling_rate = upscale_rate
        self.height_cropped_count = 1
        self.width_cropped_count = 1
        self.odd_count_list = [[0, 0]]

    def do_slicing_images(self, image_list):
        image_lists = list()
        for i in image_list:
            print("클래스 첫 진입 : ", i.shape)
            i = batch_and_transpose(i, 'CHW_TO_HWC')

            # H W C
            height, width, _ = i.shape
            print("자르기 전 변환 : ", i.shape)
            # 1000 이하 자르기 작동하지 않음
            if height <= 1000 and width <= 1000:
                # 가로와 세로의 크기가 1000 이하인 경우 자르지 않음 (테스트 중)
                image_lists.append(i)
                self.height_cropped_count = 1
                self.width_cropped_count = 1
                image_lists, self.odd_count_list = make_even_image(image_lists)
            else:
                # 가로 세로 2씩 총 4등분 필요에 따라 다른 사이즈로 정의 가능
                div_height_size = height // 2
                div_width_size = width // 2
                # 가로나 세로의 크기가 2로 나누어도 1900보다 클 경우 3으로 나눔
                if div_height_size >= 1900:
                    div_height_size = height // 3
                if div_width_size >= 1900:
                    div_width_size = width // 3
                # 이미지 자르고 리스트를 저장
                (
                    slicing_img_list,
                    self.height_cropped_count,
                    self.width_cropped_count,
                    self.odd_count_list
                ) = _image_slicing(i, div_height_size, div_width_size, self.padding_size)
                image_lists.extend(slicing_img_list)
        print("자른 이미지 변환 전: ", image_lists[0].shape)
        for idx, img in enumerate(image_lists):
            image_lists[idx] = batch_and_transpose(image_lists[idx], 'HWC_TO_CHW')
        print("자른 이미지 chw 변환: ", image_lists[0].shape)

        return image_lists, self.height_cropped_count * self.width_cropped_count

    def get_merge_image(self, slicing_img_list):

        for idx, img in enumerate(slicing_img_list):
            print("추론된 받은 이미지 : ", slicing_img_list[idx].shape)
            slicing_img_list[idx] = batch_and_transpose(slicing_img_list[idx], 'CHW_TO_HWC')

        print(slicing_img_list[0].shape)
        height_cropped_count = self.height_cropped_count
        width_cropped_count = self.width_cropped_count

        # shape 길이 측정
        if len(slicing_img_list[0].shape) == 4:
            is_batch_size = True
        else:
            is_batch_size = False

        # 추론 후 패딩 사이즈
        after_padding_size = self.padding_size * self.upscaling_rate
        image_location = [
            [0 for _ in range(height_cropped_count)] for _ in range(width_cropped_count)
        ]
        num = 0

        for row, image_line in enumerate(image_location):
            for col, _ in enumerate(image_line):
                odd_h = self.odd_count_list[num][0]
                odd_w = self.odd_count_list[num][1]
                print(slicing_img_list[num].shape)
                if odd_h == 1:
                    if is_batch_size:
                        slicing_img_list[num] = slicing_img_list[num][:, :-1 * self.upscaling_rate, :]
                    else:
                        slicing_img_list[num] = slicing_img_list[num][:-1 * self.upscaling_rate, :]
                if odd_w == 1:
                    if is_batch_size:
                        slicing_img_list[num] = slicing_img_list[num][:, :, :-1 * self.upscaling_rate]
                    else:
                        slicing_img_list[num] = slicing_img_list[num][:, :-1 * self.upscaling_rate]

                # 잘라낼 패딩 사이즈가 0 인 경우 그대로 반환
                # ([: -0]인 경우 사이즈가 바뀌지 않았는데 concat 에러가 뜸)
                if after_padding_size == 0:
                    image_location[row][col] = slicing_img_list[num]
                # 가장 아랫줄 오른쪽 조각 -> 그대로
                elif row == len(image_location) - 1 and col == len(image_location[row]) - 1:
                    image_location[row][col] = slicing_img_list[num]
                # 왼쪽 조각 -> 오른쪽 padding 자르지 않기
                elif col == len(image_location[row]) - 1:
                    if is_batch_size:
                        image_location[row][col] = slicing_img_list[num][:, :-after_padding_size, :]
                    else:
                        image_location[row][col] = slicing_img_list[num][:-after_padding_size, :]
                # 맨 아래 조각 -> 아래쪽 padding 자르지 않기
                elif row == len(image_location) - 1:
                    if is_batch_size:
                        image_location[row][col] = slicing_img_list[num][:, :, :-after_padding_size]
                    else:
                        image_location[row][col] = slicing_img_list[num][:, :-after_padding_size]

                # 나머지 조각 -> 오른쪽, 아래쪽 padding 자르기
                else:
                    if is_batch_size:
                        image_location[row][col] = slicing_img_list[num][
                                                   :, :-after_padding_size, :-after_padding_size
                                                   ]
                    else:
                        image_location[row][col] = slicing_img_list[num][
                                                   :-after_padding_size, :-after_padding_size
                                                   ]

                num += 1

        # 조각을 한줄로 한줄을 한장으로

        print("추론된 받은 변환 후 : ", image_location[0][0].shape)

        def concat_vh(list_2d):
            return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d])

        complate_image = concat_vh(image_location)

        complate_image = batch_and_transpose(complate_image, 'HWC_TO_CHW')

        return complate_image


# 원하는 추론 코드를 삽입
def for_test_rate(image_slice_list, r):
    for idx, img in enumerate(image_slice_list):
        image_slice_list[idx] = batch_and_transpose(image_slice_list[idx], 'CHW_TO_HWC')
        image_slice_list[idx] = cv2.resize(image_slice_list[idx], None, fx=r, fy=r, interpolation=cv2.INTER_CUBIC)
        debug_show_image(image_slice_list[idx])
        image_slice_list[idx] = batch_and_transpose(image_slice_list[idx], 'HWC_TO_CHW')

    return image_slice_list


def debug_show_image(img, force=False):
    if debug_show or force:
        cv2.imshow("r", img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main():
    # 처리 기준 H, W, C
    scale = 2
    padding_size = 6

    # 배열 1 이미지 임시
    img_list = [cv2.imread('Image_P/463671.jpg', cv2.IMREAD_UNCHANGED)]

    transpose_img = batch_and_transpose(img_list[0], 'HWC_TO_CHW')

    print(transpose_img.shape)
    # 클래스 선언(패딩 사이즈, 스케일 비율)
    chopper = ImageChopper(padding_size, scale)

    # 이미지 묶음 만큼 반복
    # 자른 이미지 얻기 (이미지), 이미지 총 조각 수 반환(트리톤 로직상 필요)
    image_slice_list, image_count = chopper.do_slicing_images([transpose_img])
    print("추론 전 : ", image_slice_list[0].shape)
    # 추론
    sr_img_list = for_test_rate(image_slice_list, scale)
    print("OUTPTU", sr_img_list[0].shape)
    # 이미지 합치기(이미지 리스트)
    complete_image = chopper.get_merge_image(sr_img_list)

    print(complete_image.shape)
    # REMOVE BATCH_SIZE
    complete_image = np.squeeze(complete_image)
    # CHW->HWC
    complete_image = np.transpose(complete_image, (1, 2, 0))
    # 합쳐진 이미지 확인
    debug_show_image(complete_image, force=True)
    if is_save:
        cv2.imwrite("result.png", complete_image)

if __name__ == '__main__':
    main()