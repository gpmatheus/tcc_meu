import data_loader
import cnn_model

def main():
    train, valid, test = data_loader.get_tf_datasets('TCIR-ATLN_EPAC_WPAC.h5', batch=32)

    model = cnn_model.build_model((64, 64, 3),)
    model.summary()

    # treina o modelo
    model.fit(train, validation_data=valid, epochs=10)

    loss, mae = model.evaluate(valid)
    print(f"Loss: {loss}, MAE: {mae}")

    model.save('result.h5')


if __name__ == '__main__':
    main()