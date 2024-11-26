import data_loader
import cnn_model
import tensorflow as tf

def main():

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print('# de GPUs disponíveis', len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train, valid, test, train_len, valid_len, test_len = data_loader.get_tf_datasets('TCIR-ATLN_EPAC_WPAC.h5', batch=32, force_split_data=True)

    model = cnn_model.build_model((64, 64, 3),)
    model.summary()

    # treina o modelo
    model.fit(
        train, 
        validation_data=valid, 
        epochs=500,
        steps_per_epoch=train_len,
        validation_steps=valid_len
    )

    loss, mae = model.evaluate(valid)
    print(f"Loss: {loss}, MAE: {mae}")

    model.save('result.h5')


if __name__ == '__main__':
    main()